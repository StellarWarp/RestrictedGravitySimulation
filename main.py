import taichi as ti
import sys
import numpy as np

ti.init(arch=ti.gpu)
vec3 = ti.math.vec3

G: ti.f32 = 1  # Gravitational constant


@ti.func
def m_sinh(x: ti.f32):
    return (ti.exp(x) - ti.exp(-x)) / 2


@ti.func
def m_cosh(x: ti.f32):
    return (ti.exp(x) + ti.exp(-x)) / 2


@ti.func
def m_asinh(x: ti.f32):
    return ti.log(x + ti.sqrt(x * x + 1))


@ti.func
def m_acosh(x: ti.f32):
    return ti.log(x + ti.sqrt(x * x - 1))


kepler_solver_iterations = 10

# kepler solver


@ti.func
def elliptic_kepler_init(M: ti.f32, e: ti.f32) -> ti.f32:
    t34 = e * e
    t35 = e * t34
    t33 = ti.cos(M)
    E = M + (-0.5 * t35 + e + (t34 + 3.0 /
                               2.0 * t33 * t35) * t33) * ti.sin(M)
    return E


@ti.func
def elliptic_kepler_eps3(M: ti.f32, E: ti.f32, e: ti.f32) -> ti.f32:
    sin = ti.sin(E)
    cos = ti.cos(E)
    nf1 = -1 + e * cos
    esin = e * sin
    nf0 = -E + esin + M
    t6 = nf0 / (0.5 * nf0 * esin / nf1 + nf1)
    e3 = nf0 / ((0.5 * sin - 1.0 / 6.0 * cos * t6) * e * t6 + nf1)
    return e3


@ti.func
def kepler_solver_E(M: ti.f32, e: ti.f32) -> ti.f32:
    E = elliptic_kepler_init(M, e)
    esp = sys.float_info.max
    for i in range(kepler_solver_iterations):
        esp = elliptic_kepler_eps3(M, E, e)
        E -= esp
    return E


@ti.func
def hyperbolic_kepler_init(M: ti.f32):
    return m_asinh(M)


@ti.func
def hyperbolic_kepler_eps3(M: ti.f32, H: ti.f32, e: ti.f32):
    esinh = e * m_sinh(H)
    ecosh = e * m_cosh(H)
    f0 = esinh - H - M
    f1 = ecosh - 1
    f2 = esinh
    f3 = ecosh
    e1 = f0 / f1
    e2 = f0 / (f1 - e1 * 0.5 * f2)
    e3 = f0 / (f1 - e2 * 0.5 * (f2 - e2 * f3 / 3))
    return e3


@ti.func
# e>1
def kepler_solver_H(M, e) -> ti.f32:
    H = hyperbolic_kepler_init(M)
    esp = sys.float_info.max
    for i in range(kepler_solver_iterations):
        esp = hyperbolic_kepler_eps3(M, H, e)
        H -= esp
    return H


@ti.func
def elliptic_params_to_angle(t: ti.f32, n: ti.f32, e: ti.f32) -> ti.f32:
    E = kepler_solver_E(t*n, e)
    return E


@ti.func
def hyperbolic_params_to_angle(t: ti.f32, n: ti.f32, e: ti.f32) -> ti.f32:
    return kepler_solver_H(t*n, e)


@ti.func
def elliptic_angle_to_vector(E: ti.f32, a: ti.f32, e: ti.f32, rotation: ti.math.mat3, cM: ti.f32):
    V = vec3(0, 0, 0)
    X = vec3(0, 0, 0)

    r = a * (1 - e * ti.cos(E))
    b = a * ti.sqrt(1 - e * e)
    X.x = ti.cos(E) * a - a * e
    X.y = ti.sin(E) * b

    v = ti.sqrt(G * cM * (2 / r - 1 / a))
    V.x = -ti.sin(E) * a
    V.y = ti.cos(E) * b
    V = V.normalized() * v

    X = rotation @ X
    V = rotation @ V
    return [X, V]


@ti.func
def hyperbolic_angle_to_vector(H: ti.f32, a: ti.f32, e: ti.f32, rotation: ti.math.mat3, cM: ti.f32) -> vec3:
    V = vec3(0, 0, 0)
    X = vec3(0, 0, 0)

    r = -a * (e * m_cosh(H) - 1)
    b = a * ti.sqrt(e * e - 1)
    X.x = m_cosh(H) * a - a * e
    X.y = -m_sinh(H) * b

    v = ti.sqrt(G * cM * (2 / r - 1 / a))
    V.x = m_sinh(H) * a
    V.y = -m_cosh(H) * b
    V = V.normalized() * v

    X = rotation @ X
    V = rotation @ V
    return [X, V]


@ti.func
def elliptic_params_to_vector(a: ti.f32, e: ti.f32, rotation: ti.math.mat3,
                              n: ti.f32, t: ti.f32, cM: ti.f32):
    E = elliptic_params_to_angle(t, n, e)
    return elliptic_angle_to_vector(E, a, e, rotation, cM)


@ti.func
def hyperbolic_params_to_vector(a: ti.f32, e: ti.f32, rotation: ti.math.mat3,
                                n: ti.f32, t: ti.f32, cM: ti.f32):
    H = hyperbolic_params_to_angle(t, n, e)
    return hyperbolic_angle_to_vector(H, a, e, rotation, cM)


@ti.func
def vector_to_params(X: vec3, V: vec3, cM: ti.f32, time: ti.f32):
    a: ti.f32 = 0
    e: ti.f32 = 0
    i: ti.f32 = 0
    w: ti.f32 = 0
    W: ti.f32 = 0
    n: ti.f32 = 0
    t: ti.f32 = 0
    t0: ti.f32 = 0
    # rotation:ti.math.mat3 = [1,0,0,0,1,0,0,0,1]

    X_magnitude = ti.Vector.norm(X)

    mu = cM * G
    Ws = V.dot(V) / 2 - mu / X_magnitude
    a = -mu / (2 * Ws)
    L = ti.Vector.cross(X, V)
    L_magnitude = L.norm()
    L2 = L_magnitude * L_magnitude
    p = L2 / mu
    e = ti.sqrt(1 - p / a)
    if (ti.abs(a) < 1e-6):
        e = 1

    E: ti.f32 = 0
    if (e == 0):
        E = 0
    elif (e <= 1):
        #cosE = (1 - X_magnitude / SOP.a) / SOP.e
        #sinE = Vector3.Dot(X, V) / (SOP.e * ti.sqrt(SOP.a * mu))
        cosE = (1 - X_magnitude / a)
        sinE = X.dot(V) / ti.sqrt(a * mu)
        E = ti.math.atan2(sinE, cosE)
    else:
        coshE = (1 + X_magnitude / a) / e
        sinhE = X.dot(V) / (e * ti.sqrt(-a * mu))
        #E = MathF.Atanh(sinhE / coshE)
        E = m_asinh(sinhE)

    # T = 2 * math.pi * ti.sqrt(a * a * a / mu)
    M: ti.f32 = 0
    if (e <= 1):
        M = (E - e * ti.sin(E))
    else:
        M = e * m_sinh(E) - E
    n = ti.sqrt(mu / ti.abs(a * a * a))
    t = M / n
    t0 = time - t

    sini: ti.f32 = 0
    cosi: ti.f32 = 0
    if (e != 1):

        sini = ti.sqrt(L.x * L.x + L.y * L.y) / L_magnitude
        cosi = L.z / L_magnitude
        i = ti.math.atan2(sini, cosi)
    else:
        i = 0
        sini = 0
        cosi = 1

    AscendingAxis = vec3(-L.y, L.x, 0)
    if (cosi == 1):

        AscendingAxis.x = 1
        AscendingAxis.y = 0
    elif (cosi == -1):

        AscendingAxis.x = -1
        AscendingAxis.y = 0

    θ: ti.f32 = 0  # 真近点角
    bDa2 = (1 - e * e)  # b/a^2
    if (e <= 1):
        # tanθ = √(1-e^2)sinE/(cosE-e)
        θ = ti.math.atan2(ti.sqrt(bDa2) * ti.sin(E), ti.cos(E) - e)
    else:

        sin = ti.sqrt(-bDa2) * m_sinh(E)
        cos = -(m_cosh(E) - e)
        θ = ti.math.atan2(sin, cos)

    #v = ω + θ
    # not sinv but sinv*X*A
    normalAxis = L.normalized()
    if (normalAxis.x == 0 and normalAxis.y == 0 and normalAxis.z == 0):

        normalAxis.x = 0
        normalAxis.y = 0
        normalAxis.z = 1
    sinv = ti.Vector.cross(AscendingAxis, X).dot(normalAxis)
    cosv = AscendingAxis.dot(X)
    v = ti.math.atan2(sinv, cosv)
    w = v - θ  # 近地点与参考轴的夹角

    if (i == 0):
        W = 0
    else:
        W = ti.math.atan2(AscendingAxis.y, AscendingAxis.x)

    rotation4 = ti.math.rotation3d(
        0.0, 0.0, W)@ti.math.rotation3d(i, 0.0, 0.0)@ti.math.rotation3d(0.0, 0.0, w)
    rotation = ti.math.mat3([[rotation4[0, 0], rotation4[0, 1], rotation4[0, 2]],
                             [rotation4[1, 0], rotation4[1, 1], rotation4[1, 2]],
                             [rotation4[2, 0], rotation4[2, 1], rotation4[2, 2]]])
    return Orbit(a, e, i, w, W, n, t, t0, cM, rotation)


# 0: Sun
# EllipticObjetcCount = 0
# HyperbolicObjetcCount = 0
# EllipticField = ti.field(dtype=Orbit, shape=field_end[2])
# HyperbolicField = ti.field(dtype=Orbit, shape=field_end[2])

@ti.dataclass
class Orbit:
    a: ti.f32
    e: ti.f32
    i: ti.f32
    w: ti.f32
    W: ti.f32
    n: ti.f32
    t: ti.f32
    t0: ti.f32
    M: ti.f32
    rotation: ti.math.mat3


@ti.dataclass
class ObjectVector:
    X: vec3
    V: vec3
    m: ti.f32
    r: ti.f32
    massiveFlag: ti.i32
    center: ti.i32


@ti.dataclass
class VectorLerp:
    X1: vec3
    X2: vec3
    V1: vec3
    V2: vec3
    t0: ti.f32
    t1: ti.f32


     
capacity = 100000
MassiveIndex = ti.field(dtype=ti.i32, shape=capacity)
field_end = ti.field(dtype=ti.i32, shape=3)
orbit_field = Orbit.field(shape=capacity)
vector_field = ObjectVector.field(shape=capacity)


@ti.func
def field_swap(_from, _to):
    orbit_field[_from], orbit_field[_to] = orbit_field[_to], orbit_field[_from]
    vector_field[_from], vector_field[_to] = vector_field[_to], vector_field[_from]


@ti.func
def static_add() -> ti.i32:
    field_swap(field_end[1], field_end[2])
    field_swap(field_end[0], field_end[1])
    field_end[0] = field_end[0] + 1
    field_end[1] = field_end[1] + 1
    field_end[2] = field_end[2] + 1
    return field_end[0]-1


@ti.func
def ellicptic_add() -> ti.i32:
    field_swap(field_end[1], field_end[2])
    field_end[1] = field_end[1] + 1
    field_end[2] = field_end[2] + 1
    return field_end[1]-1


@ti.func
def hyperbolic_add() -> ti.i32:
    field_end[2] = field_end[2] + 1
    return field_end[2]-1


@ti.func
def ellicptic_swap(h_index: ti.i32):
    field_swap(h_index, field_end[1])
    field_end[1] = field_end[1] + 1


@ti.func
def hyperbolic_swap(e_index: ti.i32):
    field_end[1] = field_end[1] - 1
    field_swap(e_index, field_end[1])


@ti.kernel
def initialize(count: ti.i32):
    for i in range(count):
        vector_field[i] = ObjectVector(
            vec3(ti.randn(), ti.randn(), ti.randn()).normalized()*2,
            vec3(ti.randn(), ti.randn(), ti.randn()).normalized()*2,
            m=1, r=1, massiveFlag=0, center=0)
    static_index = static_add()
    MassiveIndex[static_index] = 0
    vector_field[static_index].X = vec3(0, 0, 0)
    vector_field[static_index].V = vec3(0, 0, 0)
    vector_field[static_index].massiveFlag = 1
    vector_field[static_index].m = 10
    vector_field[static_index].r = 10
    ti.loop_config(serialize=True)
    for i in range(field_end[0], count):
        vecs = vector_field[field_end[2]]
        orbit = vector_to_params(
            vecs.X, vecs.V, vector_field[vecs.center].m, 0.0)
        if (orbit.e <= 1):
            orbit_field[ellicptic_add()] = orbit
        elif (orbit.e > 1):
            orbit_field[hyperbolic_add()] = orbit


@ti.kernel
def update_object_params(begin: ti.i32, end: ti.i32, time: ti.f32):
    for i in range(begin, end):
        vecs = vector_field[i]
        orbit_field[i] = vector_to_params(
            vecs.X, vecs.V, vector_field[vecs.center].m, time)
        if (orbit_field[i].e <= 1 and i >= field_end[1]):
            ellicptic_swap(i)
        elif (orbit_field[i].e > 1 and i < field_end[1]):
            hyperbolic_swap(i)


@ti.kernel
def update_object_vectors(time: ti.f32):

    for i in range(field_end[0], field_end[1]):
        orb = orbit_field[i]
        orb.t = time - orb.t0
        vector_field[i].X, vector_field[i].V = elliptic_params_to_vector(
            a=orb.a, e=orb.e, rotation=orb.rotation, n=orb.n, t=orb.t, cM=orb.M)
        
    for i in range(field_end[1], field_end[2]):
        orb = orbit_field[i]
        orb.t = time - orb.t0
        vector_field[i].X, vector_field[i].V = hyperbolic_params_to_vector(
            a=orb.a, e=orb.e, rotation=orb.rotation, n=orb.n, t=orb.t, cM=orb.M)


@ti.kernel
def dynamic_update(begin: ti.i32, end: ti.i32, dt: ti.f32):
    vf = ti.static(vector_field)
    for i in range(begin, end):
        cv = vf[vf[i].center]
        r = vf[i].X-cv.X
        a = -cv.m*G/(r.norm_sqr()) * r.normalized()
        vf[i].V += a*dt
        vf[i].X += vf[i].V * dt


@ti.kernel
def add_test():
    static_add()


delta_time = 0.001
time_scale:float = 1.0
simulate_time = 0

initialize(10000)
update_object_params(field_end[0], field_end[2], simulate_time)


# gui framework
window = ti.ui.Window("Restricted Gravity", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.0, 0.0, 10)
camera.lookat(0.0, 0.0, 0)

distance = 10.0


def camera_update(window, dt):
    global distance
    camera.position(0.0, distance, 1)
    camera.lookat(0.0, 0.0, 0.0)
    if (window.is_pressed(' ')):
        distance = 10.0
    if (window.is_pressed('w')):
        distance -= 100*delta_time
    if (window.is_pressed('s')):
        distance += 100*delta_time


while window.running:
    # physics
    simulate_time += delta_time*time_scale
    if (window.is_pressed('x')):
        time_scale *= 1.01
    if (window.is_pressed('z')):
        time_scale *= 0.99
    gui = window.get_gui()
    with gui.sub_window("name", 0, 0, 0.5, 0.3):
        gui.text('w and s to zoom')
        gui.text('space to reset zoom')
        gui.text('x and z to change time scale')
        gui.text(f'time_scale:{time_scale}')
    update_object_vectors(simulate_time)
    # for i in range(10):
    #       U.dynamic_update(1,U.field_end[2],delta_time/10)
    # camera
    # camera.position(0.0,0.0, 10)
    camera_update(window, delta_time)
    scene.set_camera(camera)

    # render
    scene.point_light(pos=(0, 0, 0), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    # object render
    scene.particles(centers=vector_field.X, radius=0.005, color=(1, 1, 1))

    # gui = window.get_gui()
    # with gui.sub_window("name", 0, 0, 0.5, 0.3):
    #     gui.text(content=f'X: {U.VectorField[0].X.x}')
    # new_color = gui.color_edit_3("name", old_color)

    # of1 = orbit_field[10]
    # print('a: ', of1.a, 'e: ', of1.e, 'i: ', of1.i, 'w: ', of1.w, 'W: ',
    #       of1.W, 'n: ', of1.n, 't: ', of1.t, 't0: ', of1.t0, 'M: ', of1.M)
    # print(vector_field[10].X)
    # print("field_end[0]: ", field_end[0], "field_end[1]: ", field_end[1],"field_end[2]: ", field_end[2])
    canvas.scene(scene)
    window.show()
