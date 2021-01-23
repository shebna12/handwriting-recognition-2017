import pygame

screen = pygame.display.set_mode((800,600))
pygame.display.set_caption('A bit Smudgy')
# Image(Surface) which will be refrenced
canvas = pygame.Surface((800, 600))

# Camera rectangles for sections of  the canvas
p1_camera = pygame.Rect(0,0,400,300)
p2_camera = pygame.Rect(400,0,400,300)
p3_camera = pygame.Rect(0,300,400,300)
p4_camera = pygame.Rect(400,300,400,300)

# subsurfaces of canvas
# Note that subx needs refreshing when px_camera changes.
sub1 = canvas.subsurface(p1_camera)
sub2 = canvas.subsurface(p2_camera)
sub3 = canvas.subsurface(p3_camera)
sub4 = canvas.subsurface(p4_camera)
# 

# Drawing a line on each split "screen" 
pygame.draw.line(sub2, (255,255,255), (0,0), (0,300), 10)
pygame.draw.line(sub4, (255,255,255), (0,0), (0,300), 10)
pygame.draw.line(sub3, (255,255,255), (0,0), (400,0), 10)
pygame.draw.line(sub4, (255,255,255), (0,0), (400,0), 10)

# draw player 1's view  to the top left corner
screen.blit(sub1, (0,0))
# player 2's view is in the top right corner
screen.blit(sub2, (400, 0))
# player 3's view is in the bottom left corner
screen.blit(sub3, (0, 300))
# player 4's view is in the bottom right corner
screen.blit(sub4, (400, 300))

# Update the screen
pygame.display.update()