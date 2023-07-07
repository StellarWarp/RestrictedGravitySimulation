import taichi as ti
ti.init(arch=ti.gpu)


# precompute

warp_mode_clamp: ti.i8 = 0
warp_mode_repeat: ti.i8 = 1
warp_mode_repeat_periodic: ti.i8 = 2


@ti.data_oriented
class lookup_table_2d:
    def __init__(self, func: ti.template(),x_range: tuple, y_range: tuple, size: tuple, 
                 warp_mode=(warp_mode_clamp, warp_mode_clamp),
                 periodic_offset=(0.0, 0.0)):
        self.x_begin = x_range[0]
        self.x_end = x_range[1]
        self.y_begin = y_range[0]
        self.y_end = y_range[1]
        # self.x_table_size = size[0]
        # self.y_table_size = size[1]
        self.x_size_log2 = size[0].bit_length()-1
        self.y_size_log2 = size[1].bit_length()-1
        # force size to be power of 2
        self.x_table_size = 1 << self.x_size_log2
        self.y_table_size = 1 << self.y_size_log2
        self.table = ti.field(ti.f32, shape=(
            self.x_table_size, self.y_table_size))
        self.warp_mode = warp_mode
        self.periodic_offset = periodic_offset
        self.init_table(func)

    @ti.kernel
    def init_table(self, func: ti.template()):
        for i in range(self.x_table_size):
            for j in range(self.y_table_size):
                x = self.x_begin + (self.x_end-self.x_begin) * \
                    i/self.x_table_size
                y = self.y_begin + (self.y_end-self.y_begin) * \
                    j/self.y_table_size
                self.table[i, j] = func(x, y)

    @ti.func
    def lookup(self, x, y):
        # Bilinear Interpolation
        x = (x-self.x_begin)/(self.x_end-self.x_begin)*self.x_table_size
        y = (y-self.y_begin)/(self.y_end-self.y_begin)*self.y_table_size
        x0 = ti.cast(ti.floor(x), ti.i32)
        x1 = x0+1
        y0 = ti.cast(ti.floor(y), ti.i32)
        y1 = y0+1
        x = x-x0
        y = y-y0

        offset: ti.f32 = 0.0
        # if (self.warp_mode[0] == warp_mode_repeat_periodic):
        #     i: ti.i32 = ti.cast(ti.floor(x0/self.x_table_size), ti.i32)
        #     offset += self.periodic_offset[0] * i
        # if (self.warp_mode[1] == warp_mode_repeat_periodic):
        #     i: ti.i32 = ti.cast(ti.floor(y0/self.y_table_size), ti.i32)
        #     offset += self.periodic_offset[1] * i

        if (self.warp_mode[0] == warp_mode_repeat_periodic):
            # i: ti.i32 = 0
            # if x0 < 0:
            #     i = -((-x0-1) // self.x_table_size + 1)
            # else:
            #     i = x0 // self.x_table_size

            # optimize for power of 2
            i = x0 >> self.x_size_log2
            offset += self.periodic_offset[0] * i
        if (self.warp_mode[1] == warp_mode_repeat_periodic):
            # i: ti.i32 = 0
            # if y0 < 0:
            #     i = -((-y0-1) // self.y_table_size + 1)
            # else:
            #     i = y0 // self.y_table_size

            # optimize for power of 2
            i = y0 >> self.y_size_log2
            offset += self.periodic_offset[1] * i

        if (self.warp_mode[0] == warp_mode_clamp):
            x0 = ti.math.clamp(x0, 0, self.x_table_size-1)
            x1 = ti.math.clamp(x1, 0, self.x_table_size-1)
        if (self.warp_mode[1] == warp_mode_clamp):
            y0 = ti.math.clamp(y0, 0, self.y_table_size-1)
            y1 = ti.math.clamp(y1, 0, self.y_table_size-1)
        if (self.warp_mode[0] == warp_mode_repeat or
            self.warp_mode[0] == warp_mode_repeat_periodic):
            # x0 = x0 % self.x_table_size
            # x1 = x1 % self.x_table_size

            # optimize for power of 2
            x0 = x0 & (self.x_table_size-1)
            x1 = x1 & (self.x_table_size-1)

        if (self.warp_mode[1] == warp_mode_repeat or
            self.warp_mode[1] == warp_mode_repeat_periodic):
            # y0 = y0 % self.y_table_size
            # y1 = y1 % self.y_table_size

            # optimize for power of 2
            y0 = y0 & (self.y_table_size-1)
            y1 = y1 & (self.y_table_size-1)

            

        return offset + (1-x)*(1-y)*self.table[x0, y0] + x*(1-y)*self.table[x1, y0] + (1-x)*y*self.table[x0, y1] + x*y*self.table[x1, y1]