const std = @import("std");
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
    const state = try State.init(allocator, config.n_layers, config.seq_len, config.dim);
    defer state.deinit();
    const logits = try model.transformer(state, 1, 0);
    defer allocator.free(logits);
}
