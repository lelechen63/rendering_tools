import lpips


loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
def lpips_dis( x, y):
    d = loss_fn_alex(img0, img1)

