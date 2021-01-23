import pygame

pygame.init()
gameDisplay = pygame.display.set_mode((800,600))
pygame.display.set_caption('A bit Smudgy')
clock = pygame.time.Clock()


canvas = pygame.Surface((800, 600))
p1_camera = pygame.Rect(0,0,400,300)
p2_camera = pygame.Rect(400,0,400,300)
p3_camera = pygame.Rect(0,300,400,300)
p4_camera = pygame.Rect(400,300,400,300)
# draw player 1's view  to the top left corner
gameDisplay.blit(canvas, (0,0), p1_camera)
# player 2's view is in the top right corner
gameDisplay.blit(canvas, (400, 0), p2_camera)
# player 3's view is in the bottom left corner
gameDisplay.blit(canvas, (0, 300), p3_camera)
# player 4's view is in the bottom right corner
gameDisplay.blit(canvas, (400, 300), p4_camera)

# then you update the display
# this can be done with either display.flip() or display.update(), the
# uses of each are beyond this question
pygame.display.flip()
crashed = False

while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
    	    crashed = True
        print(event)
    pygame.display.update()
    clock.tick(60)
pygame.quit()      