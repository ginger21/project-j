'''
print training process
'''


import sys

def print_pro(epoch, iteration, n_iterations, 
                info_name = " Train accuracy:",
                info_val = 0,
                is_end = False):
    pro_num = iteration%20
    if not is_end:
        sys.stdout.write(' ' * 50 + '\r')
        sys.stdout.flush()
        sys.stdout.write('Epoch_'+str(epoch+1)+': '+str(iteration + 1)+'/'+ str(n_iterations)
                    +'['+'='*(pro_num+1)+'.'*(19-pro_num)+']'
                    +info_name + str(info_val) + '\r')
        sys.stdout.flush()
    else:
        sys.stdout.write('Epoch_'+str(epoch+1)+': '+str(iteration + 1)+'/'+ str(n_iterations)
                    +'['+ '='*20+info_name + str(info_val) + '\n')
        sys.stdout.flush()