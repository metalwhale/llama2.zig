const std = @import("std");
const Allocator = std.mem.Allocator;

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `RunState` struct
pub const State = struct {
    const Self = @This();
    key_cache: []f32,
    value_cache: []f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, n_layers: usize, seq_len: usize, dim: usize) !State {
        const key_cache = try allocator.alloc(f32, n_layers * seq_len * dim);
        const value_cache = try allocator.alloc(f32, n_layers * seq_len * dim);
        const state = State{ .key_cache = key_cache, .value_cache = value_cache, .allocator = allocator };
        return state;
    }

    pub fn deinit(self: Self) void {
        self.allocator.free(self.key_cache);
        self.allocator.free(self.value_cache);
    }
};
