import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from importlib import reload

import media
reload(media)
from core import (
    abscos, 
    rotation, 
    single_ray_simulate,
    single_ray_opaque_simulate,
    RefractiveIndexConst,
    make_acute,
    make_random
)
import gradient
reload(gradient)
from gradient import (
    reliable_gradient_field,
)
from simulate import simulate_from_bottom_edge, SimConfig
from smp_simulator import simulator_opaque



def plot_rays_at_interface(direction, surface_normal, n1, n2, rand=None):
    n_now, n_next = n1, n2
    surface_normal_dot_dir = direction.dot(surface_normal)
    theta_inc = np.arctan2(-direction[0], -direction[1]) - np.arctan2(surface_normal[0], surface_normal[1])
    theta_inc_correct, is_acute = make_acute(theta_inc)
    print(np.degrees(theta_inc_correct), is_acute)
    if not is_acute:
        print(f"Warning: inaccurate gradient (obtuse angle). Manually correcting normal to f{np.degrees(theta_inc_correct)}")    
        theta_inc = theta_inc_correct
    theta_through = np.arcsin(np.abs(n_now / n_next) * np.sin(theta_inc))
    assert not np.isnan(theta_through)
    rs = (n_now * abscos(theta_inc) - n_next * abscos(theta_through)) / (n_now * abscos(theta_inc) + n_next * abscos(theta_through))
    rp = (n_next * abscos(theta_inc) - n_now * abscos(theta_through)) / (n_next * abscos(theta_inc) + n_now * abscos(theta_through))
    rho = 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)
    assert rho - 1 < 1e-7

    rand = make_random(rand)
    
    print(rho)

    if rand.uniform(0, 1) < rho: 
        print("Reflection")
        new_direction = direction - 2 * (surface_normal.dot(direction)) * surface_normal
    else:
        print("Refraction")
        new_direction = rotation(-surface_normal, -theta_through)
   
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.annotate("a", xy=(0, 0), xytext=tuple(-direction), arrowprops=dict(arrowstyle="->", color="b"))
    plt.annotate("", xy=tuple(surface_normal), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="r"))
    plt.annotate("", xy=tuple(-surface_normal), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="g"))
    plt.annotate("", xy=tuple(new_direction), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="k"))
    print(np.degrees(theta_inc), np.degrees(theta_through))
    plt.show()
    return  new_direction 


def show_gradient(medium, grad):
    x, y = np.meshgrid(np.arange(medium.shape[1]), np.arange(medium.shape[0]))
    plt.imshow(medium)
    plt.quiver(x, y, grad[1], -grad[0], scale=10, width=0.002)
    plt.show()


def example_grad_plot(medium=None):
    if medium is None:
        medium = np.zeros([30, 30])
        medium[10:20, 10:20] = 1
    grad = reliable_gradient_field(medium, smooth_sigma=0.1, normalize=True)
    x, y = plt.meshgrid(np.arange(medium.shape[1]), np.arange(medium.shape[0]))
    plt.imshow(medium)
    plt.quiver(x, y, grad[1], -grad[0], scale=10, width=0.002)
    plt.show()


def example_single_ray_sim():
    import pylab as plt
    med = (plt.imread("medium.jpg")[:, :, 0] > 100).astype(float)
    grad_field = reliable_gradient_field(med, smooth_sigma=1, normalize=True)
    initial_direction = np.array([-0.5, 1]) 
    initial_coord = np.array([278, 175])
    n_void = RefractiveIndexConst(1)
    n_matter = RefractiveIndexConst(3.5)
    wvlen = 200e-9 
    import time
    t_start = time.time()
    path, itr, coord, direction = single_ray_simulate(med, initial_direction, initial_coord, 20000, n_void, n_matter, wvlen, 
    pixel_length=0.01 * wvlen,
    grad_field=grad_field,
    rand=10
    )
    t_end = time.time()
    print(f"stats: #iterations={itr}, total time(s)={t_end - t_start}")
    plt.imshow(med, cmap="gray"); 
    plt.imshow(path, alpha=0.8)
    plt.show()


def example_single_ray_opaque_sim():
    import pylab as plt
    med = media.square_medium(500, 20, 0.3, seed=1)
    grad_field = reliable_gradient_field(med, smooth_sigma=1, normalize=True)
    initial_direction = np.array([-0.5, 1]) 
    initial_coord = np.array([268, 395])
    #initial_coord = np.array([260, 478])
    n_void = RefractiveIndexConst(1)
    n_matter = RefractiveIndexConst(3.5)
    wvlen = 200e-9 
    import time
    t_start = time.time()
    path, itr, coord, direction, strike_points, power = single_ray_opaque_simulate(
        med, initial_direction, initial_coord, 20000,
        beta=0, 
        grad_field=grad_field,
        rand=10
    )
    t_end = time.time()
    print(f"stats: #iterations={itr}, total time(s)={t_end - t_start}")
    
    sidx = tuple(np.array(strike_points).astype(int).T)
    strik_img = np.zeros_like(path)
    strik_img[sidx] = 1

    plt.imshow(med, cmap="gray"); 
    plt.imshow(path, alpha=0.8)
    plt.imshow(strik_img, alpha=0.2)
    plt.show()
    return strike_points


def example_plot_refraction_1():
    d = np.array([0.5, -0.5])  # direction of ray
    s = np.array([0, 1]) # surface normal
    plot_rays_at_interface(d, s, 1, 2, rand=10)


def example_plot_refraction_2():
    d = np.array([-0.5, -0.5]) 
    s = np.array([0, 1])
    plot_rays_at_interface(d, s, 1, 2, rand=10)


def example_plot_reflection_1():
    d = np.array([0.32, -0.95])
    s = np.array([0, 1])
    plot_rays_at_interface(d, s, 1, 3, rand=5)


def example_field_simulation_mirror():
    num_rows = 300
    med = media.single_circle(num_rows=num_rows, radius=20)
    grad_field = reliable_gradient_field(med, smooth_sigma=2, normalize=True)
    initial_direction = np.array([-1, 0])
    n_void = RefractiveIndexConst(1)
    n_matter = RefractiveIndexConst(1.0972 + 6.7942j)
    pixel_length = 1e-3 / num_rows # The medium is 1mm tall/wide
    wvlen = 200e-9
    rand = np.random.RandomState(10)
    path = simulate_from_bottom_edge(
        med=med,
        initial_direction=initial_direction,
        wvlen=wvlen,
        n_void=n_void,
        n_matter=n_matter,
        pixel_length=pixel_length,
        grad_field=grad_field,
        rand=rand,
        num=10,
        propagate_norm=5
    )
    p = convolve2d(path, np.ones((10, 10)), mode="same") / 100
    plt.imshow(med, cmap="Reds"); plt.imshow(p, alpha=0.8); 
    plt.show()


def example_field_animation_mirror():
    from matplotlib import animation
    num_rows = 300
    med = media.single_circle(num_rows=num_rows, radius=20)
    grad_field = reliable_gradient_field(med, smooth_sigma=2, normalize=True)
    n_void = RefractiveIndexConst(1)
    n_matter = RefractiveIndexConst(1.0972 + 6.7942j)
    pixel_length = 1e-3 / num_rows # The medium is 1mm tall/wide
    wvlen = 200e-9
    rand = np.random.RandomState(10)
    direction_list0 = [
        np.array([-a, np.sqrt(1 - a * a)]) for a in np.linspace(0, 1, 30)
    ]
    direction_list1 = [
        np.array([-a, -np.sqrt(1 - a * a)]) for a in np.linspace(1, 0, 30)[1:]
    ]
    direction_list = direction_list0 + direction_list1 

    path_list = []    
    fig = plt.figure()
    allimgs = []
    for ix, initial_direction in enumerate(direction_list):
        path = simulate_from_bottom_edge(
            med=med,
            initial_direction=initial_direction,
            wvlen=wvlen,
            n_void=n_void,
            n_matter=n_matter,
            pixel_length=pixel_length,
            grad_field=grad_field,
            rand=rand,
            num=1,
            propagate_norm=5
        )
        p = convolve2d(path, np.ones((10, 10)), mode="same") / 100
        allimgs.append([plt.imshow(med, cmap="Reds"), plt.imshow(p, alpha=0.8)])
    
    anim = animation.ArtistAnimation(fig, allimgs, interval=200, blit=True, repeat_delay=0)
    anim.save("./anim.gif")
    plt.show()

def example_field_animation_falling_circle():
    from matplotlib import animation
    num_rows = 300
    n_void = RefractiveIndexConst(1)
    n_matter = RefractiveIndexConst(1.0972 + 6.7942j)
    pixel_length = 1e-3 / num_rows # The medium is 1mm tall/wide
    wvlen = 200e-9
    rand = np.random.RandomState(10)

    path_list = []    
    fig = plt.figure()
    allimgs = []
    center_row_list = np.linspace(0, 2 * num_rows / 3, 50)
    for ix, center_row in enumerate(center_row_list):
        med = media.single_circle_loc(num_rows, center_row, num_rows // 2, 20)
        grad_field = reliable_gradient_field(med, smooth_sigma=2, normalize=True)
        path = simulate_from_bottom_edge(
            med=med,
            initial_direction=np.array([-1, 0]),
            wvlen=wvlen,
            n_void=n_void,
            n_matter=n_matter,
            pixel_length=pixel_length,
            grad_field=grad_field,
            rand=rand,
            num=1,
            propagate_norm=5
        )
        p = convolve2d(path, np.ones((10, 10)), mode="same") / 100
        allimgs.append([plt.imshow(med, cmap="Reds"), plt.imshow(p, alpha=0.8)])
    
    anim = animation.ArtistAnimation(fig, allimgs, interval=200, blit=True, repeat_delay=0)
    anim.save("./anim.gif")
    plt.show()

def renewal_data_collection(num_rays=20000):
    med = media.square_medium(500, 20, 0.3, seed=1)
    grad_field = reliable_gradient_field(med, smooth_sigma=2, normalize=True)

    cfg = SimConfig(
        medium = med,
        initial_direction = np.array([-1.0, 0.0]),
        initial_coord = np.array([268, 395]),
        max_itr = 2000,
        n_void = RefractiveIndexConst(1),
        n_matter = RefractiveIndexConst(3.5),
        wvlen = 200e-9,
        pixel_length = 0.01 * 200e-9,
        grad_field = grad_field,
        beta = 0
    )

    sim = simulator_opaque(cfg, num_direction_bins=8)
    sim.sim_opaque(num_sim = num_rays)

    erlang_parameters = sim.fit_erlang_distributions()
    transition_matrix_opaque = sim.get_transition_probabilities()


    return transition_matrix_opaque, erlang_parameters
    

if __name__ == "__main__":
    P_matrix, erlang_data = renewal_data_collection(num_rays=20000)
    print(np.round(P_matrix, 3))
    print("\n\n\n")

    for direction_bin, params in erlang_data.items():
        print(f"direction{direction_bin}\n"
              f"k = {params['k']}\n"
              f"theta = {params['theta']:.3f}")
    example_single_ray_sim()
