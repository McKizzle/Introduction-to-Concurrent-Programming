#Lets reset the workspace.
rm(list=ls(all.names=TRUE))

# N: The processsor count
# P: Percent of code that is parallelizable
amdahl <- function(N, P) {
  f_N = 1 / ((1 - P) + P / N)
  return(f_N)
}

# N: The processor count
# P: Percent of code that is parallelizable
gustafson <- function(N, P) {
  f_N = N - (1 - P) * (N - 1)  
  return(f_N)
}

color.palette = c("black", "orange", "skyblue", "lightseagreen", "mediumpurple4", "mediumorchid1")

# 1.a) Generate Amdahl data
N <- seq(1, 100, 5)
P <- 1 - 1 / 2 ** seq(1, 6, 1)
# 1.b) Create a plot of Amdahl's Law
plot(N, amdahl(N, tail(P, n=1)), type="o", pch=20, ylab="Speedup", xlab="Processor Count", main="Amdahl's Law", col=tail(color.palette, n=1), xaxt="n")
axis(1, at=seq(1, 100, 10))
for(i in seq(1,5)) {
  lines(N, amdahl(N, P[i]), type="o", pch=20, col=color.palette[i])
}
legend("topleft", legend=sprintf("%0.2f %%", P*100), pch=20, col=color.palette, title="Parallel Portion")
dev.copy2pdf(file="amdahls-law.pdf")



# 2.b) Create a plot of Gustafson’s Law
plot(N, gustafson(N, tail(P, n=1)), type="o", pch=20, ylab="Speedup", xlab="Processor Count", main="Gustafson's Law", col=tail(color.palette, n=1), xaxt="n")
axis(1, at=seq(1, 100, 10))
for(i in seq(1, 5)) {  
  lines(N, gustafson(N, P[i]), type="o", pch=20, col=color.palette[i])
}
legend("topleft", legend=sprintf("%0.2f %%", P*100), pch=20, col=color.palette, title="Parallel Portion")
dev.copy2pdf(file="gustafsons-law.pdf")