using TrixiParticles
using OrdinaryDiffEq
#using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.2

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 4
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
# Boundary geometry and initial fluid particle positions
tank_size = (65.0, 20.0)
initial_fluid_size = tank_size

diameter = 4
Re = 200
initial_velocity = (1.0, 0.0)
nu = 1 * diameter / Re

strouhal_number = 0.198 * (1 - 19.7 / Re)
frequency = strouhal_number * initial_velocity[1] / 1

tspan = (0.0, 5.0)

fluid_density = 1000.0
sound_speed = 10initial_velocity[1]
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(false, false, true, true), velocity=initial_velocity)

hollow_sphere = SphereShape(fluid_particle_spacing, diameter / 2, (5.0, 10.0), fluid_density,
                            n_layers=4, sphere_type=RoundSphere())

filled_sphere = SphereShape(fluid_particle_spacing, diameter / 2, (5.0, 10.0), fluid_density,
                            sphere_type=RoundSphere())

# n_particles = round(Int, 0.12 / fluid_particle_spacing)
# cylinder = RectangularShape(fluid_particle_spacing, (n_particles, n_particles), (0.2 - 1.0, 0.24 - 1.0), density=fluid_density)

fluid = setdiff(tank.fluid, filled_sphere)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

viscosity = ViscosityAdami(; nu)
#density_diffusion = DensityDiffusionAntuono(fluid, delta=0.1)
#fluid_density_calculator = ContinuityDensity()
time_step=0.01
fluid_system = ImplicitIncompressibleSPHSystem(fluid, smoothing_kernel, smoothing_length, fluid_density,
                                                viscosity=viscosity,
                                                min_iterations=5,
                                                max_iterations=10,
                                                time_step=time_step)

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = PressureZeroing()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

boundary_model_cylinder = BoundaryModelDummyParticles(hollow_sphere.density,
                                                      hollow_sphere.mass,
                                                      boundary_density_calculator,
                                                      smoothing_kernel, smoothing_length,
                                                      viscosity=viscosity)

boundary_system_cylinder = BoundarySPHSystem(hollow_sphere, boundary_model_cylinder)

# ==========================================================================================
# ==== Simulation
periodic_box = PeriodicBox(min_corner=[0.0, -1.0], max_corner=[65.0, 21.0])
cell_list = FullGridCellList(min_corner=[0.0, -1.0], max_corner=[65.0, 21.0], max_points_per_cell=500)
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box)#, cell_list)

semi = Semidiscretization(fluid_system, boundary_system, boundary_system_cylinder;
                          neighborhood_search)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = SolutionSavingCallback(dt=1.0)
shifting_callback = ParticleShiftingCallback()

callbacks = CallbackSet(info_callback, saving_callback, shifting_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, SymplecticEuler(),
            dt=time_step,
            save_everystep=false, callback=callbacks);