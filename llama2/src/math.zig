const std = @import("std");
const Allocator = std.mem.Allocator;

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `accum` function
pub fn accum(a: []f32, b: []f32) void {
    for (a, b) |*ai, bi| {
        ai.* += bi;
    }
}

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `rmsnorm` function
pub fn rmsnorm(allocator: Allocator, x: []f32, weight: []f32) ![]f32 {
    const o = try allocator.alloc(f32, x.len);
    // calculate sum of squares
    var ss: f32 = 0.0;
    for (x) |xi| {
        ss += xi * xi;
    }
    ss /= @floatFromInt(x.len);
    ss += 1e-5; // TODO: Figure out what does this mean
    ss = 1.0 / std.math.sqrt(ss);
    // normalize and scale
    for (o, x, weight) |*oi, xi, wi| {
        oi.* = wi * xi * ss;
    }
    return o;
}

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `softmax` function
pub fn softmax(x: []f32) void {
    // find max value (for numerical stability)
    var max_val: f32 = x[0];
    for (x[1..]) |xi| {
        if (xi > max_val) {
            max_val = xi;
        }
    }
    // exp and sum
    var sum: f32 = 0.0;
    for (x) |*xi| {
        xi.* = @exp(xi.* - max_val);
        sum += xi.*;
    }
    // normalize
    for (x) |*xi| {
        xi.* /= sum;
    }
}

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `matmul` function
pub fn matmul(allocator: Allocator, x: []f32, w: []f32) ![]f32 {
    // W (d,n) @ x (n,) -> xout (d,)
    const d = w.len / x.len;
    const n = x.len;
    const xout = try allocator.alloc(f32, d);
    for (xout, 0..) |*oi, i| {
        oi.* = 0;
        for (w[i * n .. (i + 1) * n], x) |wi, xi| {
            oi.* += wi * xi;
        }
    }
    return xout;
}
