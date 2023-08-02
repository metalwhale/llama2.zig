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

    // See: https://github.com/karpathy/llama2.c/blob/master/run.c, `bpe_encode` function
    pub fn bpeEncode(self: Self, text: []const u8) !std.ArrayList(usize) {
        var tokens = std.ArrayList(usize).init(self.allocator);
        // first encode every individual byte in the input string
        for (0..text.len) |i| {
            const id = try self.strLookup(text[i .. i + 1]);
            try tokens.append(id);
        }
        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            var best_score = -std.math.inf(f32);
            var best_id: ?usize = null;
            var best_idx: ?usize = null;
            for (0..tokens.items.len - 1) |i| {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                const str = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{
                    self.vocabs[tokens.items[i]],
                    self.vocabs[tokens.items[i + 1]],
                });
                defer self.allocator.free(str);
                if (self.strLookup(str)) |id| {
                    if (self.vocab_scores[id] > best_score) {
                        best_score = self.vocab_scores[id];
                        best_id = id;
                        best_idx = i;
                    }
                } else |err| {
                    _ = err catch {};
                }
            }
            if (best_idx) |idx| {
                // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
                try tokens.replaceRange(idx, 1, &[_]usize{best_id.?});
                _ = tokens.orderedRemove(idx + 1);
            } else {
                break; // we couldn't find any more pairs to merge, so we're done
            }
        }
        return tokens;
    }

    // See: https://github.com/karpathy/llama2.c/blob/master/run.c, `str_lookup` function
    fn strLookup(self: Self, str: []const u8) error{VocabNotFound}!usize {
        for (self.vocabs, 0..) |vocab, i| {
            if (std.mem.eql(u8, vocab, str)) {
                return i;
            }
        }
        return error.VocabNotFound;
    }
};
