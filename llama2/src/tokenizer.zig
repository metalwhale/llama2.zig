const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Tokenizer = struct {
    const Self = @This();
    max_token_length: i32,
    vocabs: [][]u8,
    vocab_scores: []f32,
    allocator: Allocator,

    // See: https://github.com/karpathy/llama2.c/blob/master/run.c, `main` function
    pub fn init(allocator: Allocator, file_path: []const u8, vocab_size: i32) !Tokenizer {
        const file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();
        const reader = file.reader();
        const max_token_length: i32 = @bitCast(try reader.readBytesNoEof(@sizeOf(i32)));
        // TODO: Interpolate vocab size from tokenizer file
        const vocabs = try allocator.alloc([]u8, @intCast(vocab_size));
        const vocab_scores = try allocator.alloc(f32, @intCast(vocab_size));
        for (vocabs, vocab_scores) |*vocab, *score| {
            score.* = @bitCast(try reader.readBytesNoEof(@sizeOf(f32)));
            const len: i32 = @bitCast(try reader.readBytesNoEof(@sizeOf(i32)));
            vocab.* = try allocator.alloc(u8, @intCast(len));
            _ = try reader.read(vocab.*);
        }
        const tokenizer = Tokenizer{
            .max_token_length = max_token_length,
            .vocabs = vocabs,
            .vocab_scores = vocab_scores,
            .allocator = allocator,
        };
        return tokenizer;
    }

    pub fn deinit(self: Self) void {
        for (self.vocabs) |vocab| {
            self.allocator.free(vocab);
        }
        self.allocator.free(self.vocabs);
        self.allocator.free(self.vocab_scores);
    }
};
