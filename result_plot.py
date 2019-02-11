from matplotlib import pyplot as plt
import resultloss, resultpplx, resultbelu

fig = plt.figure(figsize=(9,9))




ax1=plt.subplot2grid((12, 6), (0, 0), colspan=4, rowspan=3)
plt.xlim(1,12.5)
plt.ylim(2,7)
plt.plot(12,3.58,'o',label='baseline given')
plt.plot(resultloss.loss['1-1_0.00_NO_ATTN'].keys(),resultloss.loss['1-1_0.00_NO_ATTN'].values(),lw=1,label='baseline')
plt.plot(resultloss.loss['1-1_0.20_NO_ATTN'].keys(),resultloss.loss['1-1_0.20_NO_ATTN'].values(),lw=1,label='1-1 layer drop=0.2')
plt.plot(resultloss.loss['1-1_0.50_NO_ATTN'].keys(),resultloss.loss['1-1_0.50_NO_ATTN'].values(),lw=1,label='1-1 layer drop=0.5')
plt.plot(resultloss.loss['1-1_0.00_SOFT_ATTN'].keys(),resultloss.loss['1-1_0.00_SOFT_ATTN'].values(),lw=1,label='1-1 layer atten')
plt.plot(resultloss.loss['2-3_0.00_NO_ATTN'].keys(),resultloss.loss['2-3_0.00_NO_ATTN'].values(),lw=1,label='2-3 layer')
plt.plot(resultloss.loss['2-3_0.50_NO_ATTN'].keys(),resultloss.loss['2-3_0.50_NO_ATTN'].values(),lw=1,label='2-3 layer drop=0.5')
plt.ylabel('Training loss')
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)

ax2=plt.subplot2grid((12, 6), (4, 0), colspan=4, rowspan=3)
plt.xlim(1,12.5)
plt.ylim(25,50)
plt.plot(12,42.08,'o',label='baseline given')
plt.plot(resultpplx.pplx['1-1_0.00_NO_ATTN'].keys(),resultpplx.pplx['1-1_0.00_NO_ATTN'].values(),lw=1,label='baseline')
plt.plot(resultpplx.pplx['1-1_0.20_NO_ATTN'].keys(),resultpplx.pplx['1-1_0.20_NO_ATTN'].values(),lw=1,label='1-1 layer drop=0.2')
plt.plot(resultpplx.pplx['1-1_0.50_NO_ATTN'].keys(),resultpplx.pplx['1-1_0.50_NO_ATTN'].values(),lw=1,label='1-1 layer drop=0.5')
plt.plot(resultpplx.pplx['1-1_0.00_SOFT_ATTN'].keys(),resultpplx.pplx['1-1_0.00_SOFT_ATTN'].values(),lw=1,label='1-1 layer atten')
plt.plot(resultpplx.pplx['2-3_0.00_NO_ATTN'].keys(),resultpplx.pplx['2-3_0.00_NO_ATTN'].values(),lw=1,label='2-3 layer')
plt.plot(resultpplx.pplx['2-3_0.50_NO_ATTN'].keys(),resultpplx.pplx['2-3_0.50_NO_ATTN'].values(),lw=1,label='2-3 layer drop=0.5')
plt.ylabel('Perplexity')
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)

ax3=plt.subplot2grid((12, 6), (8, 0), colspan=4, rowspan=3)
plt.xlim(1,12.5)
plt.ylim(5,30)
plt.plot(12,17.6,'o',label='baseline given')
plt.plot(resultbelu.belu['1-1_0.00_NO_ATTN'].keys(),resultbelu.belu['1-1_0.00_NO_ATTN'].values(),lw=1,label='baseline')
plt.plot(resultbelu.belu['1-1_0.20_NO_ATTN'].keys(),resultbelu.belu['1-1_0.20_NO_ATTN'].values(),lw=1,label='1-1 layer drop=0.2')
plt.plot(resultbelu.belu['1-1_0.50_NO_ATTN'].keys(),resultbelu.belu['1-1_0.50_NO_ATTN'].values(),lw=1,label='1-1 layer drop=0.5')
plt.plot(resultbelu.belu['1-1_0.00_SOFT_ATTN'].keys(),resultbelu.belu['1-1_0.00_SOFT_ATTN'].values(),lw=1,label='1-1 layer atten')
plt.plot(resultbelu.belu['2-3_0.00_NO_ATTN'].keys(),resultbelu.belu['2-3_0.00_NO_ATTN'].values(),lw=1,label='2-3 layer')
plt.plot(resultbelu.belu['2-3_0.50_NO_ATTN'].keys(),resultbelu.belu['2-3_0.50_NO_ATTN'].values(),lw=1,label='2-3 layer drop=0.5')

plt.xlabel('Epochs')
plt.ylabel('BLEU score')
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
plt.savefig('com.pdf',format='pdf')
plt.show()
