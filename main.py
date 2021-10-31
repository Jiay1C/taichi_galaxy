import taichi as ti
import random
import numpy

ti.init(ti.cuda, debug=True)

SIZE = 1000
STEP = 1000000
G = 1
SUN_MASS = 100
SUN_RADIUS = 10
SUN_NUM = 3
EARTH_MASS = 1
EARTH_RADIUS = 3
EARTH_NUM = 100


@ti.data_oriented
class Planet:
    def __init__(self, num: ti.i32, mass: ti.f32, radius: ti.f32, step: ti.i32):
        self.num = num
        self.mass = mass
        self.radius = radius
        self.step = 1 / step
        self.pos = ti.Vector.field(2, ti.f32, num)
        self.vel = ti.Vector.field(2, ti.f32, num)
        self.force = ti.Vector.field(2, ti.f32, num)
        for i in range(num):
            self.pos[i] = [random.random(), random.random()]
        print("Init Success")

    @ti.kernel
    def update_force(self):
        for i in range(self.num):
            self.force[i] = [0, 0]
        for i in range(self.num):
            for j in range(self.num):
                if i != j:
                    diff = self.pos[j] - self.pos[i]
                    self.force[i] += self.step * (G * self.mass ** 2) / ti.sqrt(diff[0] ** 2 + diff[1] ** 2) * diff

    @ti.kernel
    def update_force_external(self, ext_pos: ti.template(), ext_mass: ti.template()):
        for i in range(self.num):
            for j in range(ext_pos.shape[0]):
                diff = ext_pos[j] - self.pos[i]
                self.force[i] += self.step * (G * self.mass * ext_mass[None]) / ti.sqrt(diff[0] ** 2 + diff[1] ** 2) * diff

    @ti.kernel
    def output(self, ret_pos: ti.template(), ret_mass: ti.template()):
        for i in self.pos:
            ret_pos[i] = self.pos[i]
        ret_mass[None] = self.mass

    @ti.kernel
    def update_vel(self):
        for i in range(self.num):
            self.vel[i] += self.force[i] / self.mass

    @ti.kernel
    def update_pos(self):
        for i in range(self.num):
            self.pos[i] += self.vel[i]

    def render(self, ext_gui):
        ext_gui.circles(self.pos.to_numpy(), color=0xffffff, radius=self.radius)


def main():
    gui = ti.GUI('Three Body', (SIZE, SIZE))
    sun = Planet(num=SUN_NUM, mass=SUN_MASS, radius=SUN_RADIUS, step=STEP)
    earth = Planet(num=EARTH_NUM, mass=EARTH_MASS, radius=EARTH_RADIUS, step=STEP)
    sun_pos = ti.Vector.field(2, ti.f32, SUN_NUM)
    sun_mass = ti.field(ti.i32, ())
    while gui.running:
        sun.render(gui)
        sun.update_force()
        sun.update_vel()
        sun.update_pos()
        sun.output(sun_pos, sun_mass)
        earth.render(gui)
        earth.update_force()
        earth.update_force_external(sun_pos, sun_mass)
        earth.update_vel()
        earth.update_pos()
        gui.show()


if __name__ == "__main__":
    main()
