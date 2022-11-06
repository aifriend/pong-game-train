import pygame

from Point import Point


class GameManager:
    def __init__(self, screen_size, ball_group, paddle_group):
        self.bg_color = pygame.Color('#000000')
        self.accent_color = (151, 151, 151)
        self.basic_font = pygame.font.Font('freesansbold.ttf', 32)
        self.score_font = pygame.font.Font('resources/bit5x3.ttf', 120)
        self.screen_width, self.screen_height = screen_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.player_score = 0
        self.opponent_score = 0
        self.ball_group = ball_group
        self.paddle_group = paddle_group
        # self.middle_strip = pygame.Rect(self.screen_width / 2 - 3, 0, 6, self.screen_height)

    @staticmethod
    def draw_dashed_line(surf, color, start_pos, end_pos, width=20, dash_length=10):
        origin = Point(start_pos)
        target = Point(end_pos)
        displacement = target - origin
        length = len(displacement)
        slope = displacement / length

        for index in range(0, length // dash_length, 2):
            start = origin + (slope * index * dash_length)
            end = origin + (slope * (index + 1) * dash_length)
            pygame.draw.line(surf, color, start.get(), end.get(), width)

    def run_game(self):
        # Drawing the game objects
        self.paddle_group.draw(self.screen)
        self.ball_group.draw(self.screen)

        # Updating the game objects
        self.paddle_group.update(self.ball_group)
        self.ball_group.update(self)
        self.reset_ball()
        self.draw_score()

    def reset_ball(self):
        if self.ball_group.sprite.rect.right >= self.screen_width:
            self.opponent_score += 1
            self.ball_group.sprite.reset_ball()
        if self.ball_group.sprite.rect.left <= 0:
            self.player_score += 1
            self.ball_group.sprite.reset_ball()

    def draw_score(self):
        player_score = self.score_font.render(str(self.player_score), True, self.accent_color)
        opponent_score = self.score_font.render(str(self.opponent_score), True, self.accent_color)

        resolution = 6
        width_offset = self.screen_width / resolution
        player_score_rect = player_score.get_rect(
            center=(width_offset * 2, self.screen_height / resolution))
        opponent_score_rect = opponent_score.get_rect(
            center=(width_offset * 4, self.screen_height / resolution))

        self.screen.blit(player_score, player_score_rect)
        self.screen.blit(opponent_score, opponent_score_rect)

    def update_score(self, countdown_number):
        time_counter = self.basic_font.render(str(countdown_number), True, self.accent_color)
        time_counter_rect = time_counter.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 50))
        pygame.draw.rect(self.screen, self.bg_color, time_counter_rect)
        self.screen.blit(time_counter, time_counter_rect)

    def background(self):
        offset = 20
        self.screen.fill(self.bg_color)
        self.draw_dashed_line(
            self.screen, self.accent_color,
            start_pos=(self.screen_width / 2 - 1, offset),
            end_pos=(self.screen_width / 2 - 1, self.screen_height - offset),
            dash_length=offset)
