const std = @import("std");
const Model = @import("model.zig").Model;

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 2) {
        std.debug.print("Usage: {s} <model_file_path>\n", .{args[0]});
        std.os.exit(1);
    }
    const model_file_path = args[1];
    const model = try Model.init(allocator, model_file_path);
    defer model.deinit();
}
