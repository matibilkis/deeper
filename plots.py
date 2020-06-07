import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from misc import Prob, ps_maxlik, qval
import tensorflow as tf
matplotlib.rcParams['agg.path.chunksize'] = 10**8

def profiles_kennedy(critic, directory, history_predictions=False):
    plt.rcParams.update({'font.size': 100})
    matplotlib.rcParams['agg.path.chunksize'] = 10**8

    plt.figure(figsize=(150,150), dpi=10)


    ax1 = plt.subplot2grid((2,2),(0,0))
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax3 = plt.subplot2grid((2,2),(1,0), colspan=2)





    betas = np.arange(.1,1.1,.05)

    if history_predictions!=False:
        axs = {"0":ax1, "1":ax2}
        for ind, epi in enumerate(history_predictions.keys()):
            if epi != "final_episode_info":

                transp_epi = float(epi)/history_predictions["final_episode_info"]
                ax3.plot(betas,history_predictions[epi]["[]"] , alpha=transp_epi, linewidth=3,label=str(epi))
                for outcome in [0.,1.]:
                   for guess_index in [0.,1.]:
                       axs[str(int(outcome))].plot(betas, history_predictions[epi][str(outcome)+str(guess_index)] , alpha=transp_epi, linewidth=3,label=str(epi))#+"-g:"+str((-1)**guess_index))



    inps = np.stack([np.ones(len(betas))*critic.pad_value, betas], axis=1)
    inps = np.reshape(inps, (len(betas),1,2))
    ax3.plot(betas, np.squeeze(critic(inps)), '--', color="black", linewidth=10,label="RNN")

    axes = {"0.0":ax1, "1.0":ax2}
    for outcome in [0.,1.]:
       for guess_index in [0.,1.]:
            m=[]
            for k in tf.unstack(inps):
                m.append(tf.concat([k, np.reshape(np.array([outcome,guess_index]), (1,2))], axis=0))
            axes[str(outcome)].plot(betas, np.squeeze(critic(tf.stack(m, axis=0)))[:,1], '--', color="black", linewidth=10,label="RNN")


    ax1.plot(betas,[qval(b, 0, -1) for b in betas],c="red", linewidth=10, label="Q(n1=0,"+r'$\beta$'+"; g=-1)")
    ax1.plot(betas,[qval(b, 0, 1) for b in betas],c="blue",  linewidth=10,label="Q(n1=0,"+r'$\beta$'+"; g=1)")

    ax2.plot(betas,[qval(b, 1, -1) for b in betas],c="red", linewidth=10, label="Q(n1=0,"+r'$\beta$'+"; g=-1)")
    ax2.plot(betas,[qval(b, 1, 1) for b in betas],c="blue",  linewidth=10,label="Q(n1=0,"+r'$\beta$'+"; g=1)")

    ax3.plot(betas,ps_maxlik(betas), linewidth=7, color="red", label="P*")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel(r'$\beta$', size=400)
        ax.legend(prop={"size":120})


    plt.savefig(directory+"/kennedy_profiles")





def just_plot(rt, pt, avg_train, helstrom, policy_evaluator, directory):
    matplotlib.rc('font', serif='cm100')
    matplotlib.rcParams['agg.path.chunksize'] = 10**8

    plt.rcParams.update({'font.size': 100})
    plt.rcParams.update({"agg.path.chunksize":10**8})
    plt.figure(figsize=(150,150), dpi=10)
    ax1=plt.subplot2grid((1,2),(0,0))
    ax2=plt.subplot2grid((1,2),(0,1))
    # ax3=plt.subplot2grid((1,3),(0,2))


    T=len(rt)
    optimal = 0.8091188035649798
    optimal_beta= -0.7499999999999993

    ax1.plot(np.log10(np.arange(1,T+1)),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
    if policy_evaluator.dolinar_layers==1:
        ax1.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="Optimal 1L")
    else:
        ax1.plot(np.log10(np.arange(1,T+1)),helstrom*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="Helstrom bound" +r'$\alpha = 0.4$')
    ax1.plot(np.log10(np.arange(1,T+1)),pt, color="blue", linewidth=15, alpha=0.8, label=r'$P_t$')

    outcomes_so_far=[]
    layer=0
    betas_would_do = policy_evaluator.recorded_trajectory_tree_would_do[str(layer)][str(np.array(outcomes_so_far))]
    betas_done = policy_evaluator.recorded_trajectory_tree[str(layer)][str(np.array(outcomes_so_far))]


    ax2.plot(np.log10(np.arange(1,len(betas_would_do)+1)), betas_would_do, color="blue", linewidth=15, alpha=0.5, label="First beta would do")
    ax2.plot(np.log10(np.arange(1,len(betas_would_do)+1)), betas_done, '-', color="red", linewidth=10, alpha=0.8, label="First beta have done")

    ax2.plot(np.log10(np.arange(1, len(betas_would_do)+1)),np.ones(len(betas_would_do))*optimal_beta, color="black", linewidth=15, alpha=0.8, label="optimal-beta")
    ax2.plot(np.log10(np.arange(1, len(betas_would_do)+1)),-np.ones(len(betas_would_do))*optimal_beta, color="black", linewidth=15, alpha=0.8)#, label="optimal-beta")


    # ax3.plot(np.arange(1,len(avg_train)+1),avg_train, color="black", linewidth=5, alpha=0.8, label="Critic's loss")



    for ax in [ax1, ax2]:
        ax.legend(prop={"size":200})

    ax1.set_xticks(range(int(np.log10(len(rt)+2))))
    tticks=[]
    for k in range(int(np.log10(len(rt)+2))):
        tticks.append("10^"+str(k))
    ax1.set_xticklabels(tticks)

    #plt.tight_layout()
    plt.savefig(directory+"/learning_plot.png")
    plt.close()

def BigPlot(buffer, rt, pt, history_betas, history_betas_would_have_done, histo_preds, losses, directory):

    matplotlib.rc('font', serif='cm10')
    plt.rcParams.update({'font.size': 40})

    plt.figure(figsize=(60,60), dpi=10)
    plt.subplots_adjust(wspace=0.5, hspace=0.3)

    T=len(rt)
    ax1=plt.subplot2grid((2,4),(0,0))
    ax2=plt.subplot2grid((2,4),(1,0))
    ax3=plt.subplot2grid((2,4),(0,1))
    ax4=plt.subplot2grid((2,4),(1,1))
    ax5=plt.subplot2grid((2,4),(0,2))
    ax6=plt.subplot2grid((2,4),(1,2))
    ax7=plt.subplot2grid((2,4),(0,3))
    ax8=plt.subplot2grid((2,4),(1,3))

    optimal = max([ps_maxlik(b) for b in buffer.betas])

    ### ploting the \Rt and \Pt ###
    ax1.plot(np.log10(np.arange(1,T+1)),rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
    ax1.plot(np.log10(np.arange(1,T+1)),optimal*np.ones(T), color="black",  linewidth=15,alpha=0.5, label="optimal")
    ax1.plot(np.log10(np.arange(1,T+1)),pt, color="blue", linewidth=8, alpha=0.3, label=r'$P_t (fluctuates!)$')

    ## ploting the histogram for betas ##
    optimal_beta = buffer.betas[np.where(ps_maxlik(buffer.betas) == max(ps_maxlik(buffer.betas)))[0][0]]
    ax2.hist(history_betas,bins=100, facecolor='r', alpha=0.6, edgecolor='blue', label="done")
    ax2.hist(history_betas_would_have_done,bins=100, facecolor='g', alpha=0.4, edgecolor='black', label="would have done")
    ax2.text(optimal_beta, 0, "*", size=30)
    ax2.text(-optimal_beta, 0, "*", size=30)

    ## ploting the history of betas ##
    ax3.plot(np.arange(1, len(history_betas)+1),history_betas, color="red", linewidth=15, alpha=0.8, label="done")
    ax3.plot(np.arange(1, len(history_betas)+1),history_betas_would_have_done, color="green", linewidth=15, alpha=0.8, label="would have done")
    ax3.plot(np.arange(1, len(history_betas)+1),np.ones(len(history_betas))*optimal_beta, color="black", linewidth=15, alpha=0.8, label="optimal-beta")
    ax3.plot(np.arange(1, len(history_betas)+1),-np.ones(len(history_betas))*optimal_beta, color="black", linewidth=15, alpha=0.8)#, label="optimal-beta")


    # #### in here i plot the loss for the first Q(0), the test and the train. Notice they have different scale! I use different colors!
    # c=0
    # lab = ["train","test"]
    # colors = ["tab:red","tab:blue"]
    # for loss in losses[0]:
    #     color = colors[c]
    loss_train = losses[0]
    ax4.plot(np.arange(1,len(loss_train)+1),loss_train,'--',alpha=0.85,c="red", linewidth=5)#, label="Preds Q(" + r'$\beta$'+")-"+lab[c])#,
     # label="Q(n1=0,"+r'$\beta$'+"; g=-1)")

    loss_train = losses[1]
    ax4.plot(np.arange(1,len(loss_train)+1),loss_train,'--',alpha=0.85,c="blue", linewidth=5)#, label="Preds Q(" + r'$\beta$'+")-"+lab[c])#,
    #     ax4.scatter(np.arange(1,len(loss)+1),loss,s=150,alpha=0.85,c=colors[c], linewidth=5)#,label="Preds Q(\beta) - "+lab[c])#, label="Q(n1=0,"+r'$\beta$'+"; g=-1)")
    #     ax4.set_xlabel("epoch", size=120)
    #     ax4.set_ylabel("Loss Q("+r'$\beta )$', size=100, color =colors[c])
    #     ax4.tick_params(axis='y', labelcolor=colors[c])
    #     ax4.legend()
    #     ax4 = ax4.twinx()  # instantiate a second axes that shares the same x-axis
    #     c+=1

    # #### in here i plot the loss for the first Q(\beta, n, guess ), the test and the train. Notice they have different scale! I use different colors!
    # c=0
    # for loss in losses[1]:
    #     ax5.plot(np.arange(1,len(loss)+1),loss,'--',alpha=0.85,c=colors[c], linewidth=5)#, label="Preds Q(n, "+r'$\beta$'+", guess) - "+lab[c])#, label="Q(n1=0,"+r'$\beta$'+"; g=-1)")
    #     ax5.scatter(np.arange(1,len(loss)+1),loss,s=150,alpha=0.85,c=colors[c], linewidth=5)#, label="Preds Q(n, \beta, guess) - "+lab[c])#, label="Q(n1=0,"+r'$\beta$'+"; g=-1)")
    #     ax5.set_xlabel("epoch", size=120)
    #     ax5.set_ylabel("Loss Q(" + r'$\beta$'+", n, guess)",size=20, color =colors[c])
    #     ax5.tick_params(axis='y', size=100, labelcolor=colors[c])
    #     #ax5.legend()
    #     ax5 = ax5.twinx()  # instantiate a second axes that shares the same x-axis
    #     c+=1
    #     #plt.tight_layout()  # otherwise the right y-label is slightly clipped
    #

    betas_train = buffer.betas
    for predictions in histo_preds["layer1"].values():
        ax7.plot(betas_train,predictions["values"]["0"],alpha=0.25, linewidth=5)#, label="epoch: "+str(predictions["epoch_number"])) #, label=r'$\hat{Q}$'+"(n1=0,"+r'$\beta$'+"; g=-1)")
        ax7.plot(betas_train,predictions["values"]["1"],alpha=0.25, linewidth=5)#, label="epoch: "+str(predictions["epoch_number"]))#,label=r'$\hat{Q}$'+"(n1=0,"+r'$\beta$'+"; g=1)")

        ax8.plot(betas_train,predictions["values"]["2"] ,alpha=0.25,  linewidth=5)#, label="epoch: "+str(predictions["epoch_number"]))#label=r'$\hat{Q}$'+"(n1=1,"+r'$\beta$'+"; g=-1)")
        ax8.plot(betas_train,predictions["values"]["3"] ,alpha=0.25,  linewidth=5)#, label="epoch: "+str(predictions["epoch_number"]))#,label=r'$\hat{Q}$'+"(n1=1,"+r'$\beta$'+"; g=1)")

    #Now we take the last and plot it in bold!
    ax7.plot(betas_train,predictions["values"]["0"],alpha=0.85, c="black",linewidth=8)#), label="epoch: "+str(predictions["epoch_number"])) #, label=r'$\hat{Q}$'+"(n1=0,"+r'$\beta$'+"; g=-1)")
    ax7.plot(betas_train,predictions["values"]["1"],alpha=0.85, c="purple", linewidth=8)#, label="epoch: "+str(predictions["epoch_number"]))#,label=r'$\hat{Q}$'+"(n1=0,"+r'$\beta$'+"; g=1)")
    ax7.scatter(betas_train,predictions["values"]["0"],alpha=0.85, c="black",s=150)
    ax7.scatter(betas_train,predictions["values"]["1"],alpha=0.85, c="purple",s=150)

    ax8.plot(betas_train,predictions["values"]["2"] ,alpha=0.85, c="black", linewidth=8)#, label="epoch: "+str(predictions["epoch_number"]))#label=r'$\hat{Q}$'+"(n1=1,"+r'$\beta$'+"; g=-1)")
    ax8.plot(betas_train,predictions["values"]["3"] ,alpha=0.85,  c="purple",linewidth=8)#, label="epoch: "+str(predictions["epoch_number"]))#,label=r'$\hat{Q}$'+"(n1=1,"+r'$\beta$'+"; g=1)")
    ax8.scatter(betas_train,predictions["values"]["2"],alpha=0.85, c="black",s=150)
    ax8.scatter(betas_train,predictions["values"]["3"],alpha=0.85, c="purple",s=150)


        ### we do the same for ax3:

    for predictions in histo_preds["layer0"].values():
        ax6.plot(betas_train,predictions["values"],alpha=0.15, linewidth=5)#, label="epoch: "+str(predictions["epoch_number"])) #, label=r'$\hat{Q}$'+"(n1=0,"+r'$\beta$'+"; g=-1)")

    #The last one black and bigger!
    ax6.plot(betas_train,predictions["values"],alpha=0.85,c="black", linewidth=5)#, label="epoch: "+str(predictions["epoch_number"])) #, label=r'$\hat{Q}$'+"(n1=0,"+r'$\beta$'+"; g=-1)")


    ##### here we plot the true values (that we want to learn!!!) ###
    ax7.plot(buffer.betas,[qval(b, 0, -1) for b in buffer.betas],'--',alpha=0.85,c="red", linewidth=8, label="Q(n1=0,"+r'$\beta$'+"; g=-1)")
    ax7.plot(buffer.betas,[qval(b, 0, 1) for b in buffer.betas],'--',alpha=0.85,c="blue",  linewidth=8,label="Q(n1=0,"+r'$\beta$'+"; g=1)")

    ax8.plot(buffer.betas,[qval(b, 1, -1) for b in buffer.betas],'--',alpha=0.85,c="red",  linewidth=8,label="Q(n1=1,"+r'$\beta$'+"; g=-1)")
    ax8.plot(buffer.betas,[qval(b, 1, 1) for b in buffer.betas],'--',alpha=0.85,c="blue",  linewidth=8,label="Q(n1=1,"+r'$\beta$'+"; g=1)")

    ax6.plot(buffer.betas,[ps_maxlik(b) for b in buffer.betas],'--',alpha=0.85,c="red", linewidth=8)
    ax6.set_ylabel(r'$P_s\; ( \beta )$', size=20)
    ##### here we plot the true values (that we want to learn!!!) ###



    for ax in [ax6, ax7, ax8]:
        ax.set_xlabel(r'$\beta$', size=20)

    for ax in [ax1, ax2, ax3,ax4,ax5,ax6, ax7, ax8]:
        ax.legend()

    #plt.tight_layout()
    plt.savefig(directory+"/big_plot.png")
    plt.close()
    return




def plot_inside_buffer(buffer, directory):
    matplotlib.rc('font', serif='cm10')
    plt.rcParams.update({'font.size': 40})

    plt.figure(figsize=(15,10))
    ax1 =  plt.subplot2grid((1,2),(0,0))
    ax2 =  plt.subplot2grid((1,2),(0,1))

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    histo = {}
    number = {}

    data_collected = np.asarray(buffer.buffer)
    for k in data_collected[:,0]:
        for g in [-1.,1.]:
            for outcome in [0.,1.]:

                histo[str(np.round(k,2))+"n"+str(outcome)+"g"+str(g)] = 0
                number[str(np.round(k,2))+"n"+str(outcome)+"g"+str(g)] = 1

    for dato in data_collected:
        histo[str(np.round(dato[0],2))+"n"+str(dato[1])+"g"+str(dato[2])] += dato[3]
        number[str(np.round(dato[0],2))+"n"+str(dato[1])+"g"+str(dato[2])] += 1

    for k in data_collected[:,0]:
        for g in [-1.,1.]:
            for outcome in [0.,1.]:
                histo[str(np.round(k,2))+"n"+str(outcome)+"g"+str(g)] /=number[str(np.round(k,2))+"n"+str(outcome)+"g"+str(g)]



    betas  = [np.round(b,2) for b in data_collected[:,0]]
    ax1.plot(betas,[histo[str(np.round(b,2))+"n0.0g-1.0"] for b in data_collected[:,0]],alpha=0.5,c="red", linewidth=5, label="Q(n1=0,"+r'$\beta$'+"; g=-1)")
    ax1.plot(betas,[histo[str(np.round(b,2))+"n0.0g1.0"] for b in data_collected[:,0]],alpha=0.5,c="blue", linewidth=5, label="Q(n1=0,"+r'$\beta$'+"; g=-1)")

    ax2.plot(betas,[histo[str(np.round(b,2))+"n1.0g-1.0"] for b in data_collected[:,0]],alpha=0.5,c="red", linewidth=5, label="Q(n1=1,"+r'$\beta$'+"; g=-1)")
    ax2.plot(betas,[histo[str(np.round(b,2))+"n1.0g1.0"] for b in data_collected[:,0]],alpha=0.5,c="blue", linewidth=5, label="Q(n1=1,"+r'$\beta$'+"; g=-1)")

    betas = np.arange(-1.5,1.5,.01)
    ax1.plot(betas,[qval(b, 0, -1) for b in betas],'--',alpha=0.5,c="red", linewidth=5, label="True Q(n1=0,"+r'$\beta$'+"; g=-1)")
    ax1.plot(betas,[qval(b, 0, 1) for b in betas],'--',alpha=0.5,c="blue",  linewidth=5,label="True Q(n1=0,"+r'$\beta$'+"; g=1)")

    ax2.plot(betas,[qval(b, 1, -1) for b in betas],'--',alpha=0.5,c="red",  linewidth=5,label="True Q(n1=1,"+r'$\beta$'+"; g=-1)")
    ax2.plot(betas,[qval(b, 1, 1) for b in betas],'--',alpha=0.5,c="blue",  linewidth=5,label="True Q(n1=1,"+r'$\beta$'+"; g=1)")


    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$\beta$', size=20)
        ax.legend(prop={"size":15})

    plt.savefig(directory+"/inside_buffer.png")
    plt.close()
    return
