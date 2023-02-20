import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.font_manager as fm
from torchvision import transforms
import numpy as np

def plot_angle(ax, x, y, angle, style):
    phi = np.radians(angle)
    xx = [x + .5, x, x + .5*np.cos(phi)]
    yy = [y, y, y + .5*np.sin(phi)]
    ax.plot(xx, yy, lw=12, color='tab:blue', solid_joinstyle=style)
    ax.plot(xx, yy, lw=1, color='black')
    ax.plot(xx[1], yy[1], 'o', color='tab:red', markersize=3)

def rotate(angle,x,y):
    phi = np.radians(angle)
    return x * np.cos(phi) - y * np.sin(phi), x * np.sin(phi) + y * np.cos(phi)

def make_polygon(n,rot,right):
    xy = np.zeros((n,2))
    angle = 360/n
    if right:
        xy[0,0] = -np.sqrt(0.5)
        xy[0,1] = 0
    else:
        xy[0,0] = np.sqrt(0.5)
        xy[0,1] = 0
    for i in range(1,n):
        x_next, y_next = rotate(angle,xy[i-1,0],xy[i-1,1])
        xy[i,0] = x_next
        xy[i,1] = y_next

    #edge_r = np.random.uniform(0,1,n)
    #edge_r = edge_r/sum(edge_r) 
    sign_index = np.random.permutation(np.concatenate([np.repeat(1,np.ceil((n)/2)),
                                          np.repeat(-1,np.ceil((n)/2))]))
    for i in range(1,n):
        x_next, y_next = rotate(sign_index[i]*rot,xy[i,0],xy[i,1])
        xy[i,0] = x_next
        xy[i,1] = y_next
        #xy[i,0] = x_next*(1+edge_r[i]*len_rnd)
        #xy[i,1] = y_next*(1+edge_r[i]*len_rnd)
    return xy

def preprocess(image,len_axis):
    #image = Image.open(image_path, 'r')
    w, h = image.size
    diff_w=len_axis-w
    diff_h=len_axis-h

    top=int(diff_w/2)
    bottom=diff_w-int(diff_w/2)
    left=int(diff_h/2)
    right=diff_h-int(diff_h/2)

    image = transforms.Pad((top,left, bottom, right), fill=(255,255,255))(image)
    #image.save("TestPentagonImages/"+os.path.basename(image_path.replace("jpg","png")))
    return image

def simulate_pentagon(n=5,
                      pentagon_size = 0.9,
                      rot = 0,
                      lw = 2,
                      dist = 1,
                      rot_right= 0,
                      size_right=1,
                      rot_both=0,
                      line_randomness = 0.1):
    # make right side shape
    xy_right = make_polygon(n,rot,True)
    

    # rotate pentagon and change size
    left_end_point = np.min(xy_right[:,0])
    for i in range(0,n):
        x_next, y_next = rotate(rot_right,xy_right[i,0],xy_right[i,1])
        xy_right[i,0] = x_next*size_right
        xy_right[i,1] = y_next*size_right
    left_end_point_after = np.min(xy_right[:,0])

    # distance between two pentagons
    xy_right[:,0]=xy_right[:,0]+dist-(left_end_point_after-left_end_point)

    # make left side shape
    xy_left = make_polygon(n,rot,False)

    # rotate whole image
    for i in range(0,n):
        x_next, y_next = rotate(rot_both,xy_right[i,0],xy_right[i,1])
        xy_right[i,0] = x_next
        xy_right[i,1] = y_next
        x_next, y_next = rotate(rot_both,xy_left[i,0],xy_left[i,1])
        xy_left[i,0] = x_next
        xy_left[i,1] = y_next

    # make fig
    plt.xkcd(scale=1, length=100, randomness=line_randomness)
    fig = plt.Figure(figsize=(4, 4))
    canvas = FigureCanvas(fig)
    
    ax = fig.gca()

    # draw right pentagon
    for i in range(1,n):
        ax.plot(xy_right[(i-1):(i+1),0], xy_right[(i-1):(i+1),1], lw=lw, color='black')
    ax.plot(xy_right[[0,n-1],0], xy_right[[0,n-1],1], lw=lw, color='black')

    # draw left pentagon
    for i in range(1,n):
        ax.plot(xy_left[(i-1):(i+1),0], xy_left[(i-1):(i+1),1], lw=lw, color='black')
    ax.plot(xy_left[[0,n-1],0], xy_left[[0,n-1],1], lw=lw, color='black')

    ax.set_aspect('equal')
    ax.margins(x=0.1,y=0.1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis('off')

    margins = {  #     vvv margin in inches
        "left"   :     0,
        "bottom" :     0,
        "right"  : 1,
        "top"    : 1
    }
    fig.subplots_adjust(**margins)
    #XKCDify(ax)
    # get image as np.array
    #canvas = fig.figure.canvas

    axis_lim = ax.axis()
    x_length = axis_lim[1]-axis_lim[0]
    y_length = axis_lim[3]-axis_lim[2]
    max_length = np.max([y_length,x_length])

    fig.set_size_inches([max_length*1.2*pentagon_size,max_length*1.2*pentagon_size])

    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    #return data.shape
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    
    indx = (np.sum(np.sum(image!=255,0)==0,1)==0) | (np.sum(np.sum(image!=255,1)==0,1)==0)
    image = image[:,indx,:]
    image = image[indx,:,:]
    
    image = Image.fromarray(image, 'RGB')
    len_axis=500
    image=preprocess(image,len_axis)
    #if display_image:
    #    img = Image.fromarray(image, 'RGB')
    #    img.show()
    
    return image