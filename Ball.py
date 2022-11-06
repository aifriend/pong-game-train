import random

import pygame

from Block import Block


class Ball(Block):
    def __init__(self, path, screen_size, offset, x_pos, y_pos, speed_x, speed_y, paddles):
        super().__init__(path, x_pos, y_pos)
        self.screen_width, self.screen_height = screen_size
        self.speed_x = speed_x * random.choice((-1, 1))
        self.speed_y = speed_y * random.choice((-1, 1))
        self.offset = offset
        self.paddles = paddles
        self.score_time = 0
        self.plob_sound = pygame.mixer.Sound("resources/pong.ogg")
        self.score_sound = pygame.mixer.Sound("resources/score.ogg")

    def update(self, game_manager):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        self.collisions()

    def collisions(self):
        if self.rect.top <= 0 + self.offset or \
                self.rect.bottom >= self.screen_height - self.offset:
            pygame.mixer.Sound.play(self.plob_sound)
            self.speed_y *= -1

        if pygame.sprite.spritecollide(self, self.paddles, False):
            pygame.mixer.Sound.play(self.plob_sound)
            collision_paddle = pygame.sprite.spritecollide(self, self.paddles, False)[0].rect
            if abs(self.rect.right - collision_paddle.left) < 10 and self.speed_x > 0:
                self.speed_x *= -1
            if abs(self.rect.left - collision_paddle.right) < 10 and self.speed_x < 0:
                self.speed_x *= -1
            if abs(self.rect.top - collision_paddle.bottom) < 10 and self.speed_y < 0:
                self.rect.top = collision_paddle.bottom
                self.speed_y *= -1
            if abs(self.rect.bottom - collision_paddle.top) < 10 and self.speed_y > 0:
                self.rect.bottom = collision_paddle.top
                self.speed_y *= -1

    def reset_ball(self):
        self.speed_x *= random.choice((-1, 1))
        self.speed_y *= random.choice((-1, 1))
        self.score_time = pygame.time.get_ticks()
        self.rect.center = (self.screen_width / 2, self.screen_height / 2)
        pygame.mixer.Sound.play(self.score_sound)
