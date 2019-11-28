from __future__ import division

import torch
import math
from PIL import ImageDraw, Image
import sys

class Logger(object):
    def __init__(self, File):
        Type = File.split('.')[-1] 
        if Type == 'error':
            self.terminal = sys.stderr
        elif Type == 'log':
            self.terminal = sys.stdout
        self.log = open(File, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

def print_args(func):
    def wrapper(*args, **kwargs):
        print(f"Arguments for function {func.__name__}")
        print("Position arguments:")
        for i, arg in enumerate(args):
            print(f"[{i}]: {arg}")
        print("Keyword arguments:")
        for arg_name, arg in kwargs.items():
            print(f"[{arg_name}]: {arg}")
        return func(*args, **kwargs)
    return wrapper

def calculate_padding(orig_height, orig_width, new_height, new_width):
    # recalculate the padding
    if max(orig_height, orig_width) == orig_height:
        new_img_width = orig_height * new_width / new_height
        scale_factor = new_height / orig_height
        pad_h = 0
        pad_w = int((new_img_width - orig_width) / 2)
    else:
        scale_factor = new_width / orig_width
        new_img_height = orig_width * new_height / new_width
        pad_w = 0
        pad_h = int((new_img_height - orig_height) / 2)
    return pad_h, pad_w, scale_factor

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def average_precision(tp, conf, n_gt):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        n_gt:  Number of ground truth objects. Always positive
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    _, i = torch.sort(-conf)
    tp, conf = tp[i].type(torch.float), conf[i].type(torch.float)

    # Accumulate FPs and TPs
    fpc = torch.cumsum(1 - tp, dim=0)
    tpc = torch.cumsum(tp, dim=0)

    # Recall
    recall_curve = tpc / (n_gt + 1e-16)
    r = (tpc[-1] / (n_gt + 1e-16))

    # Precision
    precision_curve = tpc / (tpc + fpc)
    p = tpc[-1] / (tpc[-1] + fpc[-1])

    # AP from recall-precision curve
    ap = compute_ap(recall_curve, precision_curve)

    return ap, r, p

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = torch.cat((torch.zeros((1, ), device=recall.device, dtype=recall.dtype),
                      recall,
                      torch.ones((1, ), device=recall.device, dtype=recall.dtype)))
    mpre = torch.cat((torch.zeros((1, ), device=precision.device, dtype=precision.dtype),
                      precision,
                      torch.zeros((1, ), device=precision.device, dtype=precision.dtype)))

    # compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = torch.nonzero(mrec[1:] != mrec[:-1])

    # and sum (\Delta recall) * prec
    ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def xywh2xyxy(x):  # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    #print("wh2xy")
    #print(x)
    #print("after wh2xy")
    #print(y)
    #print(y.size())
    return y

def xyxy2xywh(x):  # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    #print(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    #print(x[:,0])
    #print(x[:,2])
    #print(y[:,0])
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    #print(y[:,1])
    y[:, 2] = abs(x[:, 2] - x[:, 0])
    #print(y[:,2])
    y[:, 3] = abs(x[:, 3] - x[:, 1])
    #print(y[:,3])
    #print("xy2wh")
    #print(x)
    #print("after xy2wh")
    #print(y)
    #print(y.size())
    return y

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    #print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        #print('%5g %50s %9s %12g %20s %12g %12g' % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (i + 1, n_p, n_g))

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes.
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-12)

    return iou

def build_targets(target, anchors, num_anchors, num_classes, grid_size_h, grid_size_w, ignore_thres):
    n_b = target.size(0)
    n_a = num_anchors
    n_c = num_classes
    n_g_h = grid_size_h
    n_g_w = grid_size_w
    mask = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.uint8, device=target.device)
    conf_mask = torch.ones(n_b, n_a, n_g_h, n_g_w, dtype=torch.uint8, device=target.device)
    tx = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.float, device=target.device)
    ty = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.float, device=target.device)
    tw = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.float, device=target.device)
    th = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.float, device=target.device)
    tconf = torch.zeros(n_b, n_a, n_g_h, n_g_w, dtype=torch.float, device=target.device)
    tcls = torch.zeros(n_b, n_a, n_g_h, n_g_w, n_c, dtype=torch.uint8, device=target.device)

    master_mask = torch.sum(target, dim=2) > 0

    # Convert to position relative to box
    gx = target[:, :, 1] * n_g_w
    gy = target[:, :, 2] * n_g_h
    gw = target[:, :, 3] * n_g_w
    gh = target[:, :, 4] * n_g_h
    # Get grid box indices
    
    gi = gx.long()
    gj = gy.long()
    # setting the excess to the first row will ensure excess rows represent a valid row,
    # since all images have at least one target
    gi[~master_mask] = gi[:, 0].unsqueeze(1).expand(*gi.shape)[~master_mask]
    gj[~master_mask] = gj[:, 0].unsqueeze(1).expand(*gj.shape)[~master_mask]
    gx[~master_mask] = gx[:, 0].unsqueeze(1).expand(*gx.shape)[~master_mask]
    gy[~master_mask] = gy[:, 0].unsqueeze(1).expand(*gy.shape)[~master_mask]
    gw[~master_mask] = gw[:, 0].unsqueeze(1).expand(*gw.shape)[~master_mask]
    gh[~master_mask] = gh[:, 0].unsqueeze(1).expand(*gh.shape)[~master_mask]

    # Get shape of gt box
    a = torch.zeros((target.shape[0], target.shape[1], 2), dtype=torch.float, device=target.device)
    b = torch.unsqueeze(gw, -1)
    c = torch.unsqueeze(gh, -1)
    gt_box = torch.cat((a, b, c), dim=2)
    # Get shape of anchor box
    anchor_shapes = torch.cat((torch.zeros((anchors.shape[0], 2), device=target.device, dtype=torch.float), anchors), 1)
    # Calculate iou between gt and anchor shapes
    gt_box_1 = torch.unsqueeze(gt_box, 2).expand(-1, -1, anchor_shapes.shape[0], -1)
    anchor_shapes_1 = anchor_shapes.view(1, 1, anchor_shapes.shape[0], anchor_shapes.shape[1]).expand(*gt_box_1.shape)
    anch_ious = bbox_iou(gt_box_1, anchor_shapes_1).permute(0,2,1)  # put in same order as conf_mask

    # Where the overlap is larger than threshold set mask to zero (ignore)
    # when the condition is false, change the index to the (ignored) last row
    gj_mask = gj.unsqueeze(1).expand(-1, num_anchors, -1)[anch_ious > ignore_thres]
    gi_mask = gi.unsqueeze(1).expand(-1, num_anchors, -1)[anch_ious > ignore_thres]
    #print(gj_mask)
    #print("gj size")
    #print(gj_mask.size())
    #print(gj.size())
    #print(gi_mask)
    #print("gi size")
    #print(gi_mask.size())
    #print(gi.size())

    conf_mask[:, :, gj_mask, gi_mask] = 0
    # Find the best matching anchor box
    best_n = torch.argmax(anch_ious, dim=1)

    img_dim = torch.arange(0, n_b, device=target.device).view(-1, 1).expand(*best_n.shape)

    # Masks
    mask[img_dim, best_n, gj, gi] = 1
    conf_mask[img_dim, best_n, gj, gi] = 1
    # Coordinates
    tx[img_dim, best_n, gj, gi] = gx - gi.float()
    ty[img_dim, best_n, gj, gi] = gy - gj.float()
    # Width and height
    tw[img_dim, best_n, gj, gi] = torch.log(gw / anchors[best_n, 0] + 1e-16)
    th[img_dim, best_n, gj, gi] = torch.log(gh / anchors[best_n, 1] + 1e-16)
    # One-hot encoding of label
    target_label = target[:, :, 0].long()
    tcls[img_dim, best_n, gj, gi, target_label] = 1
    tconf[img_dim, best_n, gj, gi] = 1

    return mask, conf_mask, tx, ty, tw, th, tconf, tcls

def draw_labels_on_image(img, labels, boxes, label_names, label_colors):
    """
    Draws the provided labels on the input image.
    :param img : PIL image
        The input image, where labels will be drawn on.
    :param labels: tensor
        An N by 1 tensor, where N is the number of bounding boxes, and the value is the label index
    :param boxes: tensor
        An N by 4 tensor, where N is the number of bounding boxes, in the same order as labels. Bounding
         box coordinates are in the form XYXY, in pixel units.
    :param label_names: list[string]
        A list mapping label indices to label names
    :param label_colors: list[string]
        A list mapping label indices to PIL color strings
    :return: PIL image
        A new image, of the same dimension of the original image, with the boxes
    """
    new_img = img.copy()
    ctx = ImageDraw.Draw(new_img)
    #for label_i, box in zip(labels, boxes):
        #ctx.rectangle(box, outline=label_colors[label_i])
        #ctx.text((box[0], max(0, box[1] - 10)), text=label_names[label_i], fill=label_colors[label_i])
    return new_img

def visualize_and_save_to_local(img, labels, tmp_path='/tmp/img.jpg', box_color="red"):
    if labels is not None:
        draw = ImageDraw.Draw(img)
        for i in range(len(labels)):
            x0 = labels[i, 1].to('cpu').item() 
            y0 = labels[i, 2].to('cpu').item() 
            x1 = labels[i, 3].to('cpu').item() 
            y1 = labels[i, 4].to('cpu').item()
            #print(x0,y0,x1,y1)
            draw.rectangle((x0, y0, x1, y1), outline=box_color)

    img.save(tmp_path)

def upload_label_and_image_to_gcloud(img, labels, tmp_path='/tmp/img.jpg'):
    img.save(tmp_path)
    print(f'new image saved to {tmp_path}')



# Scale image size by multiplicative ratio
def scale_image(image, scale):
    width, height = image.size
    new_height = int(height * scale)
    new_width = int(width * scale)
    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return image

# Adds 0-index dimension to labels as class
def add_class_dimension_to_labels(labels):
    new_labels = torch.zeros((len(labels), 5),
            device=labels.device, dtype=labels.dtype)
    new_labels[:, 0] = 0  # class is 0 (for single class mode)
    new_labels[:, 1] = labels[:, 0]
    new_labels[:, 2] = labels[:, 1]
    new_labels[:, 3] = labels[:, 2]
    new_labels[:, 4] = labels[:, 3]
    return new_labels

# Convert bounding box format from [x, y, h, w] to [x1, y1, x2, y2]
#   where the x,y in [x, y, h, w] is the upper left corner
#   index_offset skips the 0-index class dimension if that exists
def xyhw2xyxy_corner(labels, skip_class_dimension=True):  
    new_labels = torch.zeros(labels.shape, device=labels.device, dtype=labels.dtype)
    i = 1 if skip_class_dimension else 0
    new_labels[:, 0+i] = labels[:, 0+i]
    new_labels[:, 1+i] = labels[:, 1+i]
    new_labels[:, 2+i] = labels[:, 0+i] + labels[:, 3+i]
    new_labels[:, 3+i] = labels[:, 1+i] + labels[:, 2+i]
    return new_labels

# Scales bounding box size
#   index_offset skips the 0-index class dimension if that exists
def scale_labels(labels, scale, skip_class_dimension=True):  
    new_labels = torch.zeros(labels.shape, device=labels.device, dtype=labels.dtype)
    i = 1 if skip_class_dimension else 0
    new_labels[:, 0+i] = scale*labels[:, 0+i]
    new_labels[:, 1+i] = scale*labels[:, 1+i]
    new_labels[:, 2+i] = scale*labels[:, 2+i]
    new_labels[:, 3+i] = scale*labels[:, 3+i]
    return new_labels

# Scales bounding box size of a xyxy-format box
#   index_offset skips the 0-index class dimension if that exists
def add_padding_on_each_side(labels, pad_w, pad_h, skip_class_dimension=True):  
    new_labels = torch.zeros(labels.shape, device=labels.device, dtype=labels.dtype)
    i = 1 if skip_class_dimension else 0
    new_labels[:, 0+i] = labels[:, 0+i] + pad_w
    new_labels[:, 1+i] = labels[:, 1+i] + pad_h
    new_labels[:, 2+i] = labels[:, 2+i] + pad_w
    new_labels[:, 3+i] = labels[:, 3+i] + pad_h
    return new_labels

# Counts number of patches that can be tiled:
#   Returns (# patches wide, # patches high, # patches total (wide x high),
#            inter-patch horizontal overlap, inter-patch vertical overlap)
def pre_tile_padding(img_width, img_height, patch_width, patch_height):
    vert_pad, horiz_pad = 0, 0
    if img_width < patch_width:
        horiz_pad = math.ceil((patch_width - img_width) / 2)
    if img_height < patch_height:
        vert_pad = math.ceil((patch_height - img_height) / 2)
    return vert_pad, horiz_pad

def get_patch_spacings(img_width, img_height, patch_width, patch_height):
    #print("img_width: {0}, img_height: {1}, ".format(img_width, img_height) + \
    #      "patch_width: {0}, patch_height: {1}".format(patch_width, patch_height))
    assert (img_width >= patch_width) and (img_height >= patch_height)

    horiz_num_patches = math.ceil(img_width/patch_width)
    horiz_overhang = horiz_num_patches*patch_width - img_width
    if horiz_num_patches == 1:
        horiz_offset = 0 # no offsets are possible
    else:
        horiz_offset = horiz_overhang / (horiz_num_patches - 1)

    vert_num_patches = math.ceil(img_height/patch_height)
    vert_overhang = vert_num_patches*patch_height - img_height
    if vert_num_patches == 1:
        vert_offset = 0 # no offsets are possible
    else:
        vert_offset = vert_overhang / (vert_num_patches - 1)

    total_patches = vert_num_patches * horiz_num_patches

    return (horiz_num_patches, vert_num_patches, total_patches, horiz_offset, vert_offset)

# Returns a tuple:
#   (cropped_image, (top, right, bottom, left) boundary positions)
#   for the cropped image representing the patch_index-th patch in image
#   patch_index is counted from upper left, left to right and top to bottom
def get_patch(image, patch_width, patch_height, patch_index):
    n_patches_wide, _, _, horiz_offset, vert_offset = \
        get_patch_spacings(image.size[0], image.size[1], patch_width, patch_height)

    # assumes top left is (0,0)
    row_position = patch_index % n_patches_wide
    left_edge = patch_width*row_position - horiz_offset*row_position
    right_edge = left_edge + patch_width

    col_position = math.floor(patch_index / n_patches_wide)
    top_edge = patch_height*col_position - vert_offset*col_position
    bottom_edge = top_edge + patch_height

    cropped = image.crop((left_edge,top_edge,right_edge,bottom_edge))
    boundary = (left_edge,top_edge,right_edge,bottom_edge)
    return cropped, boundary


def compute_overlap_rectangle(box, boundary):
    x1,x0 = min(box[2], boundary[2]), max(box[0], boundary[0])
    y1,y0 = min(box[3], boundary[3]), max(box[1], boundary[1])
    return x0,y0,x1,y1


def compute_area_overlap(box, boundary):
    dx = min(box[2], boundary[2]) - max(box[0], boundary[0])
    dy = min(box[3], boundary[3]) - max(box[1], boundary[1])
    if (dx>=0) and (dy>=0):
        return float(dx*dy)
    else:
        return 0


def compute_area(box):
    return float(box[2]-box[0])*(box[3]-box[1])


def check_point_in_boundary(point, boundary):
    left, top, right, bottom = boundary
    x,y = point
    return (left <= x <= right) and (top <= y <= bottom)


# Filter out boxes that are overlap the patch significantly (by percent or absolute)
#   New labels adjusted by the patch's relative coordinates
def filter_and_offset_labels(labels, boundary, overlap_threshold=0.5, area_threshold=1000,verbose=False):
    left, top, right, bottom = boundary
    new_labels = []

    for c,x0,y0,x1,y1 in labels:
        box = [x0,y0,x1,y1]
        box_area = compute_area(box)
        overlap_area = compute_area_overlap(box, boundary)
        if overlap_area/box_area > 0.5 or overlap_area > 1000:
            # new bounding box for the patch is just the overlap region
            new_x0, new_y0, new_x1, new_y1 = compute_overlap_rectangle(box, boundary)
            new_labels.append([c,new_x0-left,new_y0-top,new_x1-left,new_y1-top])

    if len(new_labels) > 0:
        return torch.tensor(new_labels)
    else:
        return torch.zeros((len(labels), 5))
