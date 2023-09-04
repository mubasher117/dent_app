
import numpy as np
import os

def scaled_width(width, source_image_width, dest_image_width):
    return int(dest_image_width/source_image_width*width)
def scaled_height(height, source_image_height, dest_image_height):
    return int(dest_image_height/source_image_height*height)

# Function to return upper and lower limits based using 3-sigma
def get_limits(depth_array, std = 2.5):
    depth_mean = depth_array.mean()
    depth_std = depth_array.std()
    
    minus_3sigma = depth_mean - std *depth_std
    plus_3sigma = depth_mean + std *depth_std
    
    upper_limit = min( plus_3sigma, depth_array.max() )
    lower_limit = max (minus_3sigma, depth_array.min() )
    
    return upper_limit,lower_limit

# generate heatmap from given depth array
def to_heatmap(depth_array):
    
    # min_depth = depth_array.min()
    # max_depth = depth_array.max()
    
        
    upper_limit, lower_limit = get_limits(depth_array)

    depth_range = upper_limit - lower_limit
#     print(depth_range)
    # div = int((depth_range/2)*1000)
#     print(div)
    div = 8
    unit_step = depth_range/div
    
    rows = depth_array.shape[0]
    columns = depth_array.shape[1]
    img = np.zeros([rows,columns,3],dtype=np.uint8)
    
    for x in range(rows):
        for y in range(columns):
            depth = depth_array[x,y]
            
            if (depth > upper_limit):
                depth = upper_limit
            elif (depth < lower_limit):
                depth = lower_limit
            
            s = int((depth - lower_limit)/unit_step)
            # print(s)
            img[x,y] = [0,255-s*(256/div),s*(256/div)]
            
    return img, unit_step, div

# extracting by taking the top left corner as the reference point
def extract_dent(ary_box, corner=0):
    """
    Extracts the dent, when provided with ary_box(box array) and corner.
    Here corner = 0, is top left corner
    corner = 1, is bottom left corner
    corner = 2, is bottom right corner
    corner = 3, is top right corner
    
    profile_diff computes the difference of each pixel depth with respect to reference corner, which is ultimately used
    to compute without_dent surface.
    Finally, difference of dent and without dent surface is dent_diff
    
    """
    
    nrows = ary_box.shape[0]
    ncols = ary_box.shape[1]
    
    # profile_diff computes the difference with respect to reference corner
    profile_diff = np.zeros([nrows,ncols],dtype=np.float32)
    if corner == 0:
        r = 0
        c = 0
        
        rmin = r+1
        rmax = nrows
        cmin = c+1
        cmax = ncols
     
    elif corner == 1:
        r = nrows - 1
        c = 0
        
        rmin = 0
        rmax = r
        cmin = c+1
        cmax = ncols
        
    elif corner == 2:
        r = nrows - 1
        c = ncols - 1
        
        rmin = 0
        rmax = r
        cmin = 0
        cmax = c
        
    elif corner == 3:
        r = 0
        c = ncols - 1
        
        rmin = r+1
        rmax = nrows
        cmin = 0
        cmax = c
    
    ref_corner = ary_box[r,c]
    
    for j in range(ncols):
        profile_diff[r,j] = ary_box[r,j] - ref_corner
    for i in range(nrows):
        profile_diff[i,c] = ary_box[i,c] - ref_corner
        
    for i in range(rmin, rmax):
        for j in range(cmin,cmax):
            profile_diff[i,j] = profile_diff[i,c] + profile_diff[r,j]
            
    without_dent = np.zeros([nrows,ncols],dtype=np.float32)
    
    for i in range(nrows):
        for j in range(ncols):
            without_dent[i,j] = ref_corner + profile_diff[i,j]
    
    dent_diff = without_dent - ary_box
    
    return dent_diff, without_dent, profile_diff

# Extracts depths difference close to zero from diff_array
def extractmin(diff_array):
    abs_diff = np.absolute(diff_array)
    
    min_index = 0
    min_value = abs_diff[0]
    for i in range(1,abs_diff.size):
        if abs_diff[i] < min_value:
            min_index = i
            min_value = abs_diff[i]
    return diff_array[min_index]

# function that computes the min possible dent from all four corners
def compute_aggregate_min_dent(ary_box):
    dent_diff0,_,_  = extract_dent(ary_box,0)
    dent_diff1,_,_  = extract_dent(ary_box,1)
    dent_diff2,_,_  = extract_dent(ary_box,2)
    dent_diff3,_,_  = extract_dent(ary_box,3)
    
    nrows = ary_box.shape[0]
    ncols = ary_box.shape[1]
    
    cum_dent = np.zeros([nrows,ncols],dtype=np.float32)
    for i in range(nrows):
        for j in range(ncols):
            diff_array = np.array([dent_diff0[i,j],
                                   dent_diff1[i,j],
                                   dent_diff2[i,j],
                                   dent_diff3[i,j]], dtype=np.float32 )
            cum_dent[i,j] = extractmin(diff_array)
    return cum_dent

# function that computes the avg dent from all four corners
def compute_cumm_avg_dent(ary_box):
    dent_diff0,_,_  = extract_dent(ary_box,0)
    dent_diff1,_,_  = extract_dent(ary_box,1)
    dent_diff2,_,_  = extract_dent(ary_box,2)
    dent_diff3,_,_  = extract_dent(ary_box,3)
    
    nrows = ary_box.shape[0]
    ncols = ary_box.shape[1]
    
    avg_dent = np.zeros([nrows,ncols],dtype=np.float32)
    for i in range(nrows):
        for j in range(ncols):
            avg_dent[i,j] = np.array([dent_diff0[i,j],
                                   dent_diff1[i,j],
                                   dent_diff2[i,j],
                                   dent_diff3[i,j]], dtype=np.float32).mean()

    return avg_dent

# return the name of the last subdirectory from a given path
def extract_last_subdirectory(file_path):
    # Normalize the path to handle different separators
    # print(file_path)
    normalized_path = os.path.normpath(file_path)
    print('normalized path',normalized_path)

    
    # Split the path into individual directories
    directories = normalized_path.split(os.path.sep)
    print(directories)
    
    # Remove any empty strings from the split result
    directories = [dir_name for dir_name in directories if dir_name]
    print(directories)
    
    if directories:
        return directories[-2]
    else:
        return None

def path_to_pts(objects):
    pts_array = np.empty(shape=[2, 2])
    polygon_coordinates = []
    try:
        print(type(objects))
        
        print(type(objects['path'][0]))
        path_list = objects['path'][0][:-1]
        # print(path_list)
        for coord in path_list:
            coord.pop(0)
            polygon_coordinates.append(coord)
            # print(coord)
        
        print(polygon_coordinates)
        
        pts_array = np.array(polygon_coordinates)
    except Exception as e:
            print("Error:", e)
    return pts_array