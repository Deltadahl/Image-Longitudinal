using Flux

x = ones(Float32, 2,2,1,3)
y = zeros(Float32, 2,2,1,3)
println("x: $x")
println("y: $y")

x[1,1,1,2] = 2f0
y[1,1,1,2] = 2f0

println(Flux.Losses.mse(x, y))
