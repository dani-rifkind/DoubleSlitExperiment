"""
Bohmian mechanics mode.
Author contribution: Itamar Zam implemented this project part.
"""

from functools import lru_cache
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.double_slit import de_broglie_wavelength

TITLE = "Bohmian Pilot-Wave Trajectories"

EXPLANATION = r"""
### the de broglie-bohm (pilot-wave) interpretation
this mode demonstrates a deterministic alternative to standard quantum mechanics. instead of treating particles as blurry probability clouds that only "collapse" into reality when measured, bohmian mechanics posits that **both the wave and the particle are physically real at all times**. 

**the intuition:** think of it like a surfer riding a wave. the particle is the surfer (a distinct, physical object with an exact, knowable location), and the "pilot wave" is the ocean swell dictating exactly where they go.

---

### the wave function
just like in standard quantum mechanics, the system is governed by the schrödinger wave function, $\Psi$. we can split this complex wave into two real parts—its amplitude and its phase:
$$ \Psi(\mathbf{x}, t) = R(\mathbf{x}, t) e^{i S(\mathbf{x}, t) / \hbar} $$
* **$R(\mathbf{x}, t)$** is the amplitude. its square ($R^2$) tells us the probability density of finding a particle there.
* **$S(\mathbf{x}, t)$** is the phase of the wave.

### the guidance equation (how it moves)
this is the core addition of bohmian mechanics. the particle doesn't move randomly; its velocity is strictly determined by the phase of the pilot wave. it literally "surfs" the gradient of the phase:
$$ \mathbf{v} = \frac{\nabla S}{m} = \frac{\hbar}{m} \text{Im} \left( \frac{\nabla \Psi}{\Psi} \right) $$
**why they never cross:** because the velocity field is single-valued at any point in space, trajectories can mathematically never cross. if two particles somehow reached the exact same microscopic coordinate, they would be swept away by the exact same current!

### the quantum potential (the "invisible walls")
why do the paths form those jagged, sweeping "terraces" instead of flying in straight lines? if you plug the polar wave function back into the schrödinger equation, a new, unique energy term emerges naturally, called the **quantum potential ($Q$)**:
$$ Q = -\frac{\hbar^2}{2m} \frac{\nabla^2 R}{R} $$
notice that the amplitude $R$ is in the denominator. whenever the wave experiences destructive interference (the dark nodes where $R \to 0$), the quantum potential $Q$ spikes to infinity, acting as a massive, invisible repulsive wall. 

as the particles fly out of the slits, they hit these invisible repulsive walls and are violently shoved away from the dark nodes and into the safe, low-energy "constructive" lanes, forming the beautiful terraces you see in the simulation.

### visualizing the pilot wave
in this simulation, the background displays the real part of the complex pilot wave. the alternating light and dark blue ripples represent the peaks and troughs of the wave propagating through space. if you look closely, you can see exactly how the particles (the traces) "surf" these ripples, flowing smoothly along the constructive channels and avoiding the destructive gaps!
"""

def _draw_double_slit_barrier(ax, screen_width, slit_sep, slit_w):
    vis_slit_separation=max(slit_sep,0.12*screen_width)
    vis_slit_w = max(slit_w, 0.045*screen_width)

    lower_slit= -vis_slit_separation/2
    upper_slit = vis_slit_separation /2

    gaps = [
        (lower_slit-vis_slit_w/2, lower_slit + vis_slit_w/2),
        (upper_slit - vis_slit_w/2,upper_slit+vis_slit_w/2),
    ]

    # extended the walls by 2.0 so they don't look cut off when zoomed out
    wall_segments= [
        (-screen_width*2.0,gaps[0][0]),
        (gaps[0][1], gaps[1][0]),
        (gaps[1][1],screen_width*2.0),
    ]

    for y_start,y_end in wall_segments:
        if y_end>y_start:
            ax.plot([0,0],[y_start,y_end],color="white",linewidth=7,solid_capstyle="butt",zorder=8)
            
    for y_start, y_end in gaps:
        ax.plot([0,0], [y_start, y_end], color="black",linewidth=11, solid_capstyle="butt",zorder=9)


@lru_cache(maxsize=64)
def _generate_exact_bohmian_trajectories(screen_dist,screen_width,slit_sep,slit_w,which_path_on,particle_count=150):
    #idealized near-field parameters
    ideal_wl=1.0
    ideal_k= 2*np.pi/ideal_wl
    ideal_d = 4.0
    ideal_w0=0.45
    rayleigh_z =np.pi*ideal_w0**2 /ideal_wl
    
    #restrict z-axis to the near-field region
    max_z_ideal= 10.0 
    step_count=400 
    
    #non-linear spacing to increase resolution near the slits
    t_vals=np.linspace(0,1, step_count)
    z_ideal_arr = max_z_ideal*(t_vals**1.3)
    z_ideal_arr[0]=1e-9

    ideal_tracks = np.zeros((particle_count,step_count))
    prob_quantiles= np.linspace(0.005, 0.995,particle_count)

    #compute non-crossing probability streamlines via cdf mapping
    for j,curr_z in enumerate(z_ideal_arr):
        w_at_z=ideal_w0*np.sqrt(1+(curr_z/rayleigh_z)**2)
        r_at_z = curr_z*(1+(rayleigh_z/curr_z)**2)
        
        x_boundary = ideal_d/2+6.0*w_at_z
        x_grid =np.linspace(-x_boundary, x_boundary,6000)
        
        x1= x_grid - ideal_d/2.0
        x2 =x_grid+ ideal_d/2.0
        
        env1=np.exp(-(x1**2)/(w_at_z**2))
        env2 = np.exp(-(x2**2)/(w_at_z**2))
        
        phase1= ideal_k*(x1**2)/(2*r_at_z)
        phase2= ideal_k*(x2**2) /(2*r_at_z)
        
        e1 =env1*np.exp(1j*phase1)
        e2= env2*np.exp(1j*phase2)
        
        if which_path_on:
            i_val= np.abs(e1)**2+np.abs(e2)**2
        else:
            i_val = np.abs(e1+e2)**2
            
        i_val =i_val/(np.sum(i_val)+1e-15)
        cdf_arr=np.cumsum(i_val)
        
        found_idx =np.searchsorted(cdf_arr, prob_quantiles)
        ideal_tracks[:,j]=x_grid[np.clip(found_idx,0,len(x_grid)-1)]

    #map ideal coordinates to ui space
    vis_d= max(slit_sep,0.12*screen_width)
    x_scaler =vis_d/ideal_d
    
    ui_paths=ideal_tracks*x_scaler
    real_z_vals =screen_dist*(z_ideal_arr/max_z_ideal)
    real_z_vals[0] =0.0

    spawn_times= np.random.uniform(0.0,0.85,particle_count)
    time_sort_idx =np.argsort(spawn_times)
    
    return real_z_vals,ui_paths[time_sort_idx],spawn_times[time_sort_idx],ui_paths[time_sort_idx,0],ui_paths[time_sort_idx,-1]


def render(params):
    src_dist=params["source_distance"]
    sc_dist = params["screen_distance"]
    sc_width=params["screen_width"]
    dist_between_slits= params["slit_distance"]
    single_slit_w =params["slit_width"]

    curr_progress= float(params.get("progress",0.0))
    curr_progress=np.clip(curr_progress, 0.0,1.0)

    z_arr,track_paths,start_times,init_x,final_x= _generate_exact_bohmian_trajectories(
        sc_dist,
        sc_width,
        dist_between_slits,
        single_slit_w,
        params["which_path"]
    )

    fly_time= 0.15 
    is_spawned=curr_progress>=start_times
    did_hit =curr_progress>=(start_times+fly_time)

    #top view animation plot
    fig_top,ax_top=plt.subplots(figsize=(9.8,5.2))
    ax_top.set_facecolor("#0a192f") 

    _draw_double_slit_barrier(ax_top,sc_width,dist_between_slits,single_slit_w)

    ax_top.scatter([-src_dist],[0],marker="*",s=220,color="white",edgecolors="black",zorder=10)
    ax_top.text(-src_dist,0.87*sc_width,"source",color="white",ha="center",fontsize=10,weight="bold")
    ax_top.axvline(sc_dist,color="red",linewidth=3.0,zorder=8)
    ax_top.text(sc_dist,0.87*sc_width,"screen",color="red",ha="center",fontsize=10,weight="bold")

    #calculate 2d background pilot wave
    ideal_wl= 1.0
    ideal_k=2*np.pi/ideal_wl
    ideal_d= 4.0
    ideal_w0=0.45  
    rayleigh_z= np.pi*ideal_w0**2/ideal_wl
    max_z_ideal= 10.0 
    
    vis_d= max(dist_between_slits,0.12*sc_width)
    x_scaler=vis_d/ideal_d
    
    bg_res=350
    z_bg_arr=np.linspace(-src_dist,sc_dist,bg_res)
    # expanded wave mesh by 1.5x to fill zoomed out frame
    x_bg_arr = np.linspace(-sc_width*1.5, sc_width*1.5, bg_res)
    z_ui_grid,x_ui_grid= np.meshgrid(z_bg_arr,x_bg_arr)
    psi_bg =np.zeros_like(z_ui_grid,dtype=complex)
    
    #this introduces the omega*t phase shift so the waves ripple outward based on slider progress
    time_phase= -curr_progress*150.0

    #region 1: interference after the slits
    mask_pos =z_ui_grid>=0
    if np.any(mask_pos):
        z_pos_id = z_ui_grid[mask_pos]*(max_z_ideal/sc_dist)
        x_pos_id=x_ui_grid[mask_pos]/x_scaler
        z_pos_id=np.maximum(z_pos_id,1e-9)
        
        w_at_z=ideal_w0*np.sqrt(1+(z_pos_id/rayleigh_z)**2)
        r_at_z = z_pos_id*(1+(rayleigh_z/z_pos_id)**2)
        x1= x_pos_id-ideal_d/2.0
        x2= x_pos_id+ ideal_d/2.0
        
        env1=np.exp(-(x1**2)/(w_at_z**2))
        env2= np.exp(-(x2**2)/(w_at_z**2))
        
        #add longitudinal phase and time phase to create the visual ripples
        phase1=ideal_k*z_pos_id+ideal_k*(x1**2)/(2*r_at_z)+time_phase
        phase2= ideal_k*z_pos_id +ideal_k*(x2**2)/(2*r_at_z)+time_phase
        
        e1 =env1*np.exp(1j*phase1)
        e2= env2*np.exp(1j*phase2)
        
        if params["which_path"]:
            psi_bg[mask_pos]=np.abs(e1)+np.abs(e2)
        else:
            psi_bg[mask_pos] =e1+e2

    #region 2: expanding source wave before the slits
    mask_neg= z_ui_grid<0
    if np.any(mask_neg):
        z_neg_id= z_ui_grid[mask_neg]*(max_z_ideal/sc_dist)
        x_neg_id =x_ui_grid[mask_neg]/x_scaler
        src_z_id = -max_z_ideal*(src_dist/sc_dist)
        
        dist_from_src= z_neg_id-src_z_id
        dist_from_src=np.maximum(dist_from_src, 1e-9)
        
        w_src= ideal_w0*3.0*np.sqrt(1+(dist_from_src/rayleigh_z)**2)
        r_src =dist_from_src*(1+(rayleigh_z/dist_from_src)**2)
        env_src =np.exp(-(x_neg_id**2)/(w_src**2))
        phase_src= ideal_k*dist_from_src+ideal_k*(x_neg_id**2)/(2*r_src)+time_phase
        
        psi_bg[mask_neg]= env_src*np.exp(1j*phase_src)

    wave_vis= np.real(psi_bg)
    norm_val=np.abs(psi_bg)+1e-9
    
    #compress the amplitude mathematically so far-field ripples are visible 
    wave_vis= wave_vis/norm_val*(norm_val**0.4)

    #expand extent to match the new 1.5x zoom
    ax_top.imshow(wave_vis,extent=[-src_dist,sc_dist,-sc_width*1.5,sc_width*1.5],origin='lower',cmap='Blues',alpha=0.6,aspect='auto',zorder=2)

    total_z_length =src_dist+sc_dist
    
    if np.any(is_spawned):
        prog_ratios=(curr_progress-start_times)/fly_time
        curr_z_positions= -src_dist+prog_ratios*total_z_length
        drawn_subset =np.linspace(0,len(init_x)-1,min(150,len(init_x))).astype(int)
        
        for i in drawn_subset:
            if not is_spawned[i]:
                continue
                
            z_now=curr_z_positions[i]
            start_x_pos=init_x[i]
            
            def get_x_at_z(z_pts):
                return np.interp(z_pts,z_arr,track_paths[i])

            #draw particle trails
            if did_hit[i]:
                z_line=np.linspace(0,sc_dist,250)
                # Changed to orange with higher alpha to contrast with the blue background
                ax_top.plot([-src_dist,0],[0,start_x_pos],color="orange",alpha=0.35,linewidth=1.0,zorder=5)
                ax_top.plot(z_line,get_x_at_z(z_line),color="orange",alpha=0.35,linewidth=1.0,zorder=5)
            else:
                if z_now<0:
                    z_line=np.linspace(-src_dist,z_now,10)
                    ax_top.plot(z_line,(z_line+src_dist)/src_dist*start_x_pos,color="#b200ff",alpha=0.9,linewidth=2.0,zorder=6)
                else:
                    ax_top.plot([-src_dist,0],[0,start_x_pos],color="#b200ff",alpha=0.9,linewidth=2.0,zorder=6)
                    z_line=np.linspace(0,z_now,150)
                    ax_top.plot(z_line,get_x_at_z(z_line),color="#b200ff",alpha=0.9,linewidth=2.0,zorder=6)
                
                #active particle dot
                if z_now>=0:
                    current_x= get_x_at_z(np.array([z_now]))[0]
                else:
                    current_x =(z_now+src_dist)/src_dist*start_x_pos
                ax_top.scatter(z_now,current_x,color="#00ffcc",s=25,zorder=11)

    if np.any(did_hit):
        landed_hits=final_x[did_hit]
        ax_top.scatter(np.full_like(landed_hits,sc_dist),landed_hits,color="orange",s=6,alpha=0.4,zorder=12)

    ax_top.set_title("Top View: Deterministic Pilot-Wave Trajectories",fontsize=13)
    ax_top.set_xlabel("Propagation direction z [m]")
    ax_top.set_ylabel("Transverse position x [m]")
    ax_top.set_xlim(-src_dist,sc_dist+0.24*sc_dist)
    
    # --- ZOOM FIX 1: Zoom out the top view by 1.5x ---
    ax_top.set_ylim(-sc_width*1.5,sc_width*1.5)

    #intensity distribution plot
    fig_screen,ax_screen=plt.subplots(figsize=(6.6,3.0))

    end_z=10.0 

    w_at_z_end=ideal_w0*np.sqrt(1+(end_z/rayleigh_z)**2)
    r_at_z_end =end_z*(1+(rayleigh_z/end_z)**2)
    x_vals_ideal= np.linspace(-100.0,100.0,4000)
    
    x1= x_vals_ideal-ideal_d/2.0
    x2=x_vals_ideal+ideal_d/2.0

    env1=np.exp(-(x1**2)/(w_at_z_end**2))
    env2 =np.exp(-(x2**2)/(w_at_z_end**2))
    phase1= ideal_k*(x1**2)/(2*r_at_z_end)
    phase2= ideal_k*(x2**2)/(2*r_at_z_end)
    
    e1=env1*np.exp(1j*phase1)
    e2=env2*np.exp(1j*phase2)

    if params["which_path"]:
        i_ideal_val=np.abs(e1)**2+np.abs(e2)**2
    else:
        i_ideal_val= np.abs(e1+e2)**2

    i_normalized =i_ideal_val/(np.max(i_ideal_val)+1e-12)
    x_vals_ui =x_vals_ideal*x_scaler

    #calculate the absolute maximum expected hit count when all particles arrive
    all_final_hits= final_x*1e3
    #expand the math range for counting by 1.5x as well
    all_counts,_ = np.histogram(all_final_hits,bins=120,range=(-sc_width*1e3*1.5,sc_width*1e3*1.5))
    max_expected_count= np.max(all_counts)

    if np.any(did_hit):
        ax_screen.plot(x_vals_ui*1e3,i_normalized,color="red",linewidth=2.0,alpha=0.5,label="Pilot-Wave Field")

        landed_hits= final_x[did_hit]*1e3 
        hist_counts,bin_edges=np.histogram(landed_hits,bins=120,range=(-sc_width*1e3*1.5,sc_width*1e3*1.5))
        
        if max_expected_count>0:
            norm_hist=hist_counts/max_expected_count
            ax_screen.bar(bin_edges[:-1],norm_hist,width=np.diff(bin_edges),align="edge",color="orange",alpha=0.75,label="Particle Arrivals")

    ax_screen.set_title("Detection Pattern on Screen")
    ax_screen.set_xlabel("Screen position x [mm]")
    ax_screen.set_ylabel("Relative Intensity")
    ax_screen.set_ylim(0,1.05)
    
    # zoom out the screen pattern by 1.5x ---
    ax_screen.set_xlim(-sc_width*1e3*1.5, sc_width*1e3*1.5) 
    
    ax_screen.grid(True,alpha=0.35)
    
    if np.any(did_hit):
        ax_screen.legend(loc="upper right")

    return fig_top,fig_screen