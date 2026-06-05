import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

print("Devices:", len(devices))

for d in devices:
    print(d.get_info(rs.camera_info.name))

ctx = rs.context()
for d in ctx.query_devices():
    print(d.get_info(rs.camera_info.serial_number))