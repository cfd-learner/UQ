import string

name = 'LBM_2D_NS_cuKernel.cu'

f = open(name,'r')
h = open('modified.cl','wt')

for line in f.readlines():
    line = 'f.write(\'' + line.rstrip('\r\n') + '\\n\')' + '\n'
    h.write(line)

f.close()
h.close()