import pygame

from Point import Point


class GameManager:
    def __init__(self, screen_size, offset, ball_group, paddle_group):
        self.offset = offset
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
        self.playground_strip_up = pygame.Rect(0, 0, self.screen_width, self.offset)
        self.playground_strip_down = pygame.Rect(0, self.screen_height - self.offset, self.screen_width, self.offset)

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

    def draw_score(self):
        player_score = self.score_font.render(str(self.player_score), True, self.accent_color)
        opponent_score = self.score_font.render(str(self.opponent_score), True, self.accent_color)

        resolution = 15
        width_offset = self.screen_width / resolution
        opponent_score_rect = opponent_score.get_rect(
            center=(width_offset * 6, self.screen_height / resolution * 1.7))
        player_score_rect = player_score.get_rect(
            center=(width_offset * 9, self.screen_height / resolution * 1.7))

        self.screen.blit(player_score, player_score_rect)
        self.screen.blit(opponent_score, opponent_score_rect)

    def background(self):
        self.screen.fill(self.bg_color)
        pygame.draw.rect(self.screen, self.accent_color, self.playground_strip_up)
        pygame.draw.rect(self.screen, self.accent_color, self.playground_strip_down)
        self.draw_dashed_line(
            self.screen, self.accent_color,
            start_pos=(self.screen_width / 2 - 1, self.offset),
            end_pos=(self.screen_width / 2 - 1, self.screen_height),
            width=self.offset - 5,
            dash_length=self.offset)
