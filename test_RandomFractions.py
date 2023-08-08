from VorticityCorrelation import test_RandomFractions, test_FDistribution

test_RandomFractions()
test_FDistribution(M=100000, T=20000, CPU=mp.cpu_count(), C=0, serial=False)
