import pygame

# 初始化Pygame
pygame.init()

# 设置窗口大小
size = (400, 300)
screen = pygame.display.set_mode(size)

# 设置窗口标题
pygame.display.set_caption("My Pygame Window")

# 主循环
running = True
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 填充背景色
    screen.fill((255, 255, 255))

    # 更新窗口
    pygame.display.flip()

# 退出Pygame
pygame.quit()