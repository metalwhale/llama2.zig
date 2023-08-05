const std = @import("std");
const Allocator = std.mem.Allocator;

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `accum` function
pub fn accum(a: []f32, b: []const f32) void {
    for (a, b) |*ai, bi| {
        ai.* += bi;
    }
}

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `rmsnorm` function
pub fn rmsnorm(x: []const f32, weight: []const f32, dest: []f32) void {
    // calculate sum of squares
    var ss: f32 = 0.0;
    for (x) |xi| {
        ss += xi * xi;
    }
    ss /= @floatFromInt(x.len);
    ss += 1e-5; // TODO: Figure out what does this mean
    ss = 1.0 / std.math.sqrt(ss);
    // normalize and scale
    for (dest, x, weight) |*d, xi, wi| {
        d.* = wi * xi * ss;
    }
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
pub fn matmul(x: []const f32, w: []const f32, dest: []f32) void {
    // W (d,n) @ x (n,) -> xout (d,)
    const n = x.len;
    for (dest, 0..) |*d, i| {
        d.* = 0;
        for (w[i * n .. (i + 1) * n], x) |wi, xi| {
            d.* += wi * xi;
        }
    }
}

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `argmax` function
pub fn argmax(v: []const f32) usize {
    var max_i: usize = 0;
    var max_p = v[0];
    for (1..v.len) |i| {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

// See: https://github.com/karpathy/llama2.c/blob/master/run.c, `argmax` function
pub fn sample(prng: *std.rand.DefaultPrng, probabilities: []f32) usize {
    // sample index from probabilities, they must sum to 1
    const r = prng.random().float(f32) * std.math.floatMax(f32);
    var cdf = 0.0;
    for (probabilities, 0..) |p, i| {
        cdf += p;
        if (r < cdf) {
            return i;
        }
    }
    // in case of rounding errors
    return probabilities.len - 1;
}

pub fn dot(a: []f32, b: []f32) f32 {
    var d: f32 = 0.0;
    for (a, b) |ai, bi| {
        d += ai * bi;
    }
    return d;
}

// F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
pub fn silu(x: []f32) void {
    for (x) |*xi| {
        xi.* = xi.* / (1.0 + std.math.exp(-xi.*));
    }
}
