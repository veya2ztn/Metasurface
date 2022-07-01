def have_a_image_snap(model,input_vector,input_image,args):
    model.G.eval()
    model.I2C.eval()
    #nrows,ncols = args.snap_size
    nrows,ncols = 2,4
    sample_num=nrows*ncols
    random_sample=np.random.choice(range(len(input_vector)),sample_num,replace=False).tolist()
    input_vector =input_vector[random_sample]
    input_image  =input_image[random_sample]
    z = get_lattened_vector(input_vector,sample_num,args) #(9,100,1,1)
    with torch.no_grad():
        fake_images = model.G(z)
        fakebinary  = ((fake_images*0.5)+0.5)
        fake_vectors= model.I2C(fakebinary)
        fakebinary  = fakebinary.round()
        pred_vectors= model.I2C(fakebinary)

    I_FAKE = fake_images[:,0].cpu().detach().numpy()
    I_REAL = input_image[:,0].cpu().detach().numpy()
    I_FAKE = ((I_FAKE*0.5)+0.5)
    I_PRED = I_FAKE.round()
    I_REAL = ((I_REAL*0.5)+0.5).round();
    real_v = input_vector[:,0].cpu().detach().numpy()
    fake_v = fake_vectors[:,0].cpu().detach().numpy()
    pred_v = pred_vectors[:,0].cpu().detach().numpy()
    score  = ((real_v-pred_v)**2).mean(1)
    data = list(zip(score,real_v,fake_v,pred_v,I_REAL,I_FAKE,I_PRED))
    fake_image_figure=show_gan_image_demo(data,nrows=nrows,ncols=ncols)
    return fake_image_figure

def plot16x16(images):
    graph_fig, graph_axes = plt.subplots(nrows=4, ncols=4, figsize=(16,16))
    graph_axes = graph_axes.flatten()
    for image,ax in zip(images,graph_axes):
        _=ax.imshow(image,cmap='hot',vmin=0, vmax=1)
        _=ax.set_xticks(())
        _=ax.set_yticks(())
    return graph_fig
