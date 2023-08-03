const std = @import("std");
const utils = @import("utils.zig");
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

pub const Model = struct {
    const Self = @This();
    config: Config,
    weight: Weight,
    memory: []align(std.mem.page_size) const u8,
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
        const model = Model{ .config = config, .weight = weight, .memory = memory, .allocator = allocator };
        return model;
    }

    pub fn deinit(self: Self) void {
        std.os.munmap(self.memory);
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
