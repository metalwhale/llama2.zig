const std = @import("std");
const math = @import("math.zig");
const Model = @import("model.zig").Model;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const State = @import("state.zig").State;

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 3) {
        std.debug.print("Usage: {s} <model_file_path> <tokenizer_file_path>\n", .{args[0]});
        std.os.exit(1);
    }
    // Read model
    const model_file_path = args[1];
    const model = try Model.init(allocator, model_file_path);
    defer model.deinit();
    const config = model.config;
    // Read tokenizer
    const tokenizer_file_path = args[2];
    const tokenizer = try Tokenizer.init(allocator, tokenizer_file_path, config.vocab_size);
    defer tokenizer.deinit();
    // process the prompt
    const prompt_tokens = try tokenizer.bpeEncode("Hello darkness, my old friend");
    defer prompt_tokens.deinit();
    // start the main loop
    var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
    const state = try State.init(allocator, config.n_layers, config.seq_len, config.dim);
    defer state.deinit();
    const vocabs = tokenizer.vocabs;
    const num_prompt_tokens = prompt_tokens.items.len;
    const temperature = 0.0; // TODO: Get this from command line
    const steps = config.seq_len; // TODO: Get this from command line
    var pos: usize = 0;
    var token: usize = 1;
    while (pos < steps) : (pos += 1) {
        var next: usize = undefined;
        // forward the transformer to get logits for the next token
        const logits = try model.transformer(state, token, 0);
        defer allocator.free(logits);
        if (pos < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens.items[pos];
        } else {
            // sample the next token
            if (temperature == 0.0) {
                // greedy argmax sampling: take the token with the highest probability
                next = math.argmax(logits);
            } else {
                // apply the temperature to the logits
                for (logits) |*l| {
                    l.* /= temperature;
                }
                // apply softmax to the logits to get the probabilities for next token
                math.softmax(logits);
                // we sample from this distribution to get the next token
                next = math.sample(prng, logits);
            }
        }
        // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
        const token_str = vocabs[next][if (token == 1 and std.mem.eql(u8, vocabs[next][0..1], " ")) 1 else 0..];
        std.debug.print("{s}", .{token_str});
        // advance forward
        token = next;
    }
}
