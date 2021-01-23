from utils import get_midpoint,sort_LR


contours = [9,5,3,1,8,4,7,2]
contours = sort_LR(contours)
print("CONTOURS: ",contours)
midpoint = get_midpoint()
print("Midpoint:",midpoint)