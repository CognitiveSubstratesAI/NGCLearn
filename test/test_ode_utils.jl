using NGCLearn: get_integrator_code, step_euler, step_rk2

@testset "ode_utils" begin
    @testset "get_integrator_code" begin
        @test get_integrator_code("euler") == 0
        @test get_integrator_code("rk1") == 0
        @test get_integrator_code("midpoint") == 1
        @test get_integrator_code("rk2") == 1
        @test get_integrator_code("heun") == 2
        @test get_integrator_code("rk2_heun") == 2
        @test get_integrator_code("ralston") == 3
        @test get_integrator_code("rk2_ralston") == 3
        @test get_integrator_code("rk4") == 4
        @test get_integrator_code("not-a-method") == 0  # falls back to Euler
    end

    @testset "step_euler exactness on linear ODE" begin
        # dx/dt = a (constant) ⇒ Euler is exact: x_next = x + a*dt
        dfx = (t, x, p) -> p[1]
        t1, x1 = step_euler(0.0, 10.0, dfx, 0.5, (4.0,))
        @test t1 == 0.5
        @test x1 ≈ 12.0
    end

    @testset "step_rk2 exact on linear, 2nd-order on quadratic" begin
        # Constant slope ⇒ both Euler and midpoint are exact.
        dfx_const = (t, x, p) -> 2.0
        _, x1 = step_rk2(0.0, 1.0, dfx_const, 1.0, ())
        @test x1 ≈ 3.0

        # dx/dt = t ⇒ exact solution x(dt) = x0 + dt^2/2. Midpoint integrates
        # this exactly; Euler does not (it sees slope 0 at t=0).
        dfx_t = (t, x, p) -> t
        _, x_rk2 = step_rk2(0.0, 0.0, dfx_t, 2.0, ())
        @test x_rk2 ≈ 2.0                # 0 + 2^2/2
        _, x_eu = step_euler(0.0, 0.0, dfx_t, 2.0, ())
        @test x_eu ≈ 0.0                 # Euler undershoots
    end

    @testset "broadcasting over arrays" begin
        dfx = (t, x, p) -> fill(1.0, size(x))
        _, x1 = step_euler(0.0, [1.0 2.0 3.0], dfx, 0.5, ())
        @test x1 ≈ [1.5 2.5 3.5]
    end
end
