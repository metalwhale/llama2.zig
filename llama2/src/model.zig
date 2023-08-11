const std = @import("std");
const utils = @import("utils.zig");
const math = @import("math.zig");
const State = @import("state.zig").State;
const Allocator = std.mem.Allocator;

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `Config` struct
const Config = struct {
    dim: usize, // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize, // number of layers
    n_heads: usize, // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    seq_len: usize, // max sequence length
};

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `TransformerWeights` struct
const Weight = struct {
    // token embedding table
    token_embedding_table: []const f32, // (vocab_size, dim)
    // attention rmsnorm
    rms_att_weight: []const f32, // (layer, dim)
    // matmuls
    wq: []const f32, // (layer, dim, dim)
    wk: []const f32, // (layer, dim, dim)
    wv: []const f32, // (layer, dim, dim)
    wo: []const f32, // (layer, dim, dim)
    // ffn rmsnorm
    rms_ffn_weight: []const f32, // (layer, dim)
    // ffn
    w1: []const f32, // (layer, hidden_dim, dim)
    w2: []const f32, // (layer, dim, hidden_dim)
    w3: []const f32, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: []const f32, // (dim)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: []const f32, // (seq_len, (dim/n_heads)/2)
    freq_cis_imag: []const f32, // (seq_len, (dim/n_heads)/2)
};

const Buffer = struct {
    const Self = @This();
    x: []f32, // (dim)
    xb: []f32, // (dim)
    xb2: []f32, // (dim)
    q: []f32, // (dim)
    k: []f32, // (dim)
    v: []f32, // (dim)
    hb1: []f32, // (hidden_dim)
    hb3: []f32, // (hidden_dim)
    att: []f32, // (seq_len)
    logits: []f32, // (vocab_size)
    allocator: Allocator,

    fn init(allocator: Allocator, config: Config) !Buffer {
        const x = try allocator.alloc(f32, config.dim);
        const xb = try allocator.alloc(f32, config.dim);
        const xb2 = try allocator.alloc(f32, config.dim);
        const q = try allocator.alloc(f32, config.dim);
        const k = try allocator.alloc(f32, config.dim);
        const v = try allocator.alloc(f32, config.dim);
        const hb1 = try allocator.alloc(f32, config.hidden_dim);
        const hb3 = try allocator.alloc(f32, config.hidden_dim);
        const att = try allocator.alloc(f32, config.seq_len);
        const logits = try allocator.alloc(f32, config.vocab_size);
        const buffer = Buffer{
            .x = x,
            .xb = xb,
            .xb2 = xb2,
            .q = q,
            .k = k,
            .v = v,
            .hb1 = hb1,
            .hb3 = hb3,
            .att = att,
            .logits = logits,
            .allocator = allocator,
        };
        return buffer;
    }

    fn deinit(self: Self) void {
        self.allocator.free(self.x);
        self.allocator.free(self.xb);
        self.allocator.free(self.xb2);
        self.allocator.free(self.q);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.allocator.free(self.hb1);
        self.allocator.free(self.hb3);
        self.allocator.free(self.att);
        self.allocator.free(self.logits);
    }
};

pub const Model = struct {
    const Self = @This();
    config: Config,
    weight: Weight,
    memory: []align(std.mem.page_size) const u8,
    buffer: Buffer,
    allocator: Allocator,

    // References:
    // - Llama architecture: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    // - Check the `export` functions from the following files to see how `.bin` model files were created:
    //   - TinyLlama: https://github.com/karpathy/llama2.c/blob/master/model.py
    //   - Llama: https://github.com/karpathy/llama2.c/blob/master/export_meta_llama_bin.py
    pub fn init(allocator: Allocator, file_path: []const u8) !Model {
        const file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();
        const reader = file.reader();
        // Read config
        const config_bytes: [7]i32 = @bitCast(try reader.readBytesNoEof(@sizeOf(i32) * 7));
        const config: Config = .{
            .dim = @intCast(config_bytes[0]),
            .hidden_dim = @intCast(config_bytes[1]),
            .n_layers = @intCast(config_bytes[2]),
            .n_heads = @intCast(config_bytes[3]),
            .n_kv_heads = @intCast(config_bytes[4]),
            .vocab_size = @intCast(config_bytes[5]),
            .seq_len = @intCast(config_bytes[6]),
        };
        const pos = try file.getPos();
        const file_size = (try file.stat()).size;
        const memory = try std.os.mmap(
            null,
            file_size,
            std.os.PROT.READ,
            std.os.MAP.PRIVATE,
            file.handle,
            0,
        );
        // Read weights
        const weight = try readWeight(allocator, config, memory[pos..]);
        // Buffer
        const buffer = try Buffer.init(allocator, config);
        const model = Model{ .config = config, .weight = weight, .memory = memory, .buffer = buffer, .allocator = allocator };
        return model;
    }

    pub fn deinit(self: Self) void {
        std.os.munmap(self.memory);
        self.buffer.deinit();
    }

    // See: https://github.com/karpathy/llama2.c/blob/master/run.c, `transformer` function
    pub fn transformer(self: Self, state: State, token: usize, pos: usize) []f32 {
        // a few convenience variables
        const dim = self.config.dim;
        const hidden_dim = self.config.hidden_dim;
        const head_size = dim / self.config.n_heads;
        const weight = self.weight;
        const buffer = self.buffer;
        // copy the token embedding into x
        std.mem.copy(f32, buffer.x, weight.token_embedding_table[token * dim .. (token + 1) * dim]);
        // forward all the layers
        for (0..self.config.n_layers) |l| {
            math.rmsnorm(buffer.x, weight.rms_att_weight[l * dim .. (l + 1) * dim], buffer.xb);
            // qkv matmuls for this position
            math.matmul(buffer.xb, weight.wq[l * dim * dim .. (l + 1) * dim * dim], buffer.q); // (dim)
            math.matmul(buffer.xb, weight.wk[l * dim * dim .. (l + 1) * dim * dim], buffer.k); // (dim)
            math.matmul(buffer.xb, weight.wv[l * dim * dim .. (l + 1) * dim * dim], buffer.v); // (dim)
            // apply RoPE rotation to the q and k vectors for each head
            rope(
                self.config,
                buffer.q,
                buffer.k,
                weight.freq_cis_real[pos * head_size / 2 .. (pos + 1) * head_size / 2],
                weight.freq_cis_imag[pos * head_size / 2 .. (pos + 1) * head_size / 2],
            );
            // save key,value at this time step (pos) to our kv cache
            const loff = l * self.config.seq_len * dim;
            std.mem.copy(f32, state.key_cache[loff + pos * dim .. loff + (pos + 1) * dim], buffer.k);
            std.mem.copy(f32, state.value_cache[loff + pos * dim .. loff + (pos + 1) * dim], buffer.v);
            // multihead attention. iterate over all heads
            attention(
                self.config,
                buffer,
                buffer.q,
                state,
                loff,
                pos,
                weight.wo[l * dim * dim .. (l + 1) * dim * dim],
            ); // (dim)
            // residual connection back into x
            math.accum(buffer.x, buffer.xb2);
            // ffn rmsnorm
            math.rmsnorm(buffer.x, weight.rms_ffn_weight[l * dim .. (l + 1) * dim], buffer.xb);
            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            math.matmul(
                buffer.xb,
                weight.w1[l * hidden_dim * dim .. (l + 1) * hidden_dim * dim],
                buffer.hb1,
            ); // (hidden_dim)
            math.matmul(
                buffer.xb,
                weight.w3[l * hidden_dim * dim .. (l + 1) * hidden_dim * dim],
                buffer.hb3,
            ); // (hidden_dim)
            math.silu(buffer.hb1);
            // elementwise multiply with w3(x)
            for (buffer.hb1, buffer.hb3) |*hb1i, hb3i| {
                hb1i.* = hb1i.* * hb3i;
            }
            // final matmul to get the output of the ffn
            math.matmul(
                buffer.hb1,
                weight.w2[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim],
                buffer.xb,
            ); // (dim)
            // residual connection
            math.accum(buffer.x, buffer.xb);
        }
        // final rmsnorm
        math.rmsnorm(buffer.x, weight.rms_final_weight, buffer.x);
        // classifier into logits
        // shared weight
        math.matmul(buffer.x, weight.token_embedding_table, buffer.logits);
        return buffer.logits;
    }
};

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `checkpoint_init_weights` function
fn readWeight(allocator: Allocator, config: Config, weight_memory: []const u8) !Weight {
    const head_size = @divExact(config.dim, config.n_heads);
    // TODO: Explore methods for calculating the size of each weights
    const weight_sizes = [_]usize{
        config.vocab_size * config.dim, // token_embedding_table
        config.n_layers * config.dim, // rms_att_weight
        config.n_layers * config.dim * config.dim, // wq
        config.n_layers * config.dim * config.dim, // wk
        config.n_layers * config.dim * config.dim, // wv
        config.n_layers * config.dim * config.dim, // wo
        config.n_layers * config.dim, // rms_ffn_weight
        config.n_layers * config.hidden_dim * config.dim, // w1
        config.n_layers * config.dim * config.hidden_dim, // w2
        config.n_layers * config.hidden_dim * config.dim, // w3
        config.dim, // rms_final_weight
        config.seq_len * @divExact(head_size, 2), // freq_cis_real
        config.seq_len * @divExact(head_size, 2), // freq_cis_imag
    };
    const memories = try allocator.alloc([]const u8, weight_sizes.len);
    defer allocator.free(memories);
    var pos: usize = 0;
    for (weight_sizes, memories) |size, *memory| {
        const next_pos = pos + size * @sizeOf(f32);
        memory.* = weight_memory[pos..next_pos];
        pos = next_pos;
    }
    const weight = Weight{
        .token_embedding_table = utils.sliceCast(f32, memories[0]),
        .rms_att_weight = utils.sliceCast(f32, memories[1]),
        .wq = utils.sliceCast(f32, memories[2]),
        .wk = utils.sliceCast(f32, memories[3]),
        .wv = utils.sliceCast(f32, memories[4]),
        .wo = utils.sliceCast(f32, memories[5]),
        .rms_ffn_weight = utils.sliceCast(f32, memories[6]),
        .w1 = utils.sliceCast(f32, memories[7]),
        .w2 = utils.sliceCast(f32, memories[8]),
        .w3 = utils.sliceCast(f32, memories[9]),
        .rms_final_weight = utils.sliceCast(f32, memories[10]),
        .freq_cis_real = utils.sliceCast(f32, memories[11]),
        .freq_cis_imag = utils.sliceCast(f32, memories[12]),
    };
    return weight;
}

fn rope(config: Config, q: []f32, k: []f32, freq_cis_real_row: []const f32, freq_cis_imag_row: []const f32) void {
    const head_size = config.dim / config.n_heads;
    for (0..config.n_heads) |h| {
        // get the q and k vectors for this head
        const qh = q[h * head_size .. (h + 1) * head_size];
        const kh = k[h * head_size .. (h + 1) * head_size];
        // rotate q and k by the freq_cis_real and freq_cis_imag
        var i: usize = 0;
        while (i < head_size) : (i += 2) {
            const q0 = qh[i];
            const q1 = qh[i + 1];
            const k0 = kh[i];
            const k1 = kh[i + 1];
            const fcr = freq_cis_real_row[i / 2];
            const fci = freq_cis_imag_row[i / 2];
            qh[i] = q0 * fcr - q1 * fci;
            qh[i + 1] = q0 * fci + q1 * fcr;
            kh[i] = k0 * fcr - k1 * fci;
            kh[i + 1] = k0 * fci + k1 * fcr;
        }
    }
}

fn attention(
    config: Config,
    buffer: Buffer,
    q: []f32,
    state: State,
    loff: usize,
    pos: usize,
    ol: []const f32,
) void {
    const dim = config.dim;
    const head_size = dim / config.n_heads;
    for (0..config.n_heads) |h| {
        // get the query vector for this head
        const qh = q[h * head_size .. (h + 1) * head_size]; // (head_size)
        // attention scores for this head
        // iterate over all timesteps, including the current one
        const att = buffer.att[0 .. pos + 1];
        for (att, 0..) |*a, t| {
            // get the key vector for this head and at this timestep
            const kh = state.key_cache[loff + t * dim + h * head_size .. loff + t * dim + (h + 1) * head_size]; // (head_size)
            // calculate the attention score as the dot product of q and k
            a.* = math.dot(qh, kh) / std.math.sqrt(@as(f32, @floatFromInt(head_size)));
        }
        // softmax the scores to get attention weights, from 0..pos inclusively
        math.softmax(att);
        // weighted sum of the values
        const xbh = buffer.xb[h * head_size .. (h + 1) * head_size]; // (head_size)
        for (xbh) |*xbhi| {
            xbhi.* = 0;
        }
        for (att, 0..) |a, t| {
            // get the value vector for this head and at this timestep
            const vh = state.value_cache[loff + t * dim + h * head_size .. loff + t * dim + (h + 1) * head_size]; // (head_size)
            // accumulate the weighted value into xb
            for (xbh, vh) |*xbhi, vhi| {
                xbhi.* += a * vhi;
            }
        }
    }
    // final matmul to get the output of the attention
    math.matmul(buffer.xb, ol, buffer.xb2);
}
