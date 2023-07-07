import taichi as ti
import numpy as np
from lookup_table import *

vec3 = ti.math.vec3

G: ti.f32 = 1  # Gravitational constant
FLOAT_MAX: ti.f32 = np.finfo(np.float32).max


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


kepler_solver_iterations = 5

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
def kepler_solver_E_iter(M: ti.f32, e: ti.f32) -> ti.f32:
    E = elliptic_kepler_init(M, e)
    esp = FLOAT_MAX
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
def kepler_solver_H_iter(M, e) -> ti.f32:
    H = hyperbolic_kepler_init(M)
    esp = FLOAT_MAX
    for i in range(kepler_solver_iterations):
        esp = hyperbolic_kepler_eps3(M, H, e)
        H -= esp
    return H


kepler_solver_E = lookup_table_2d(
    kepler_solver_E_iter, (0, 2 * np.pi), (0, 1), (128, 128),
    warp_mode=(warp_mode_repeat_periodic, warp_mode_clamp),
    periodic_offset=(2*np.pi, 0))

kepler_solver_H = lookup_table_2d(
    kepler_solver_H_iter, (0, 2 * np.pi), (0, 1), (128, 128))


@ti.func
def elliptic_params_to_angle(t: ti.f32, n: ti.f32, e: ti.f32) -> ti.f32:
    # E = kepler_solver_E.lookup(t*n, e)
    E = kepler_solver_E_iter(t*n, e)
    return E


@ti.func
def hyperbolic_params_to_angle(t: ti.f32, n: ti.f32, e: ti.f32) -> ti.f32:
    # return kepler_solver_H.lookup(t*n, e)
    return kepler_solver_H_iter(t*n, e)


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
