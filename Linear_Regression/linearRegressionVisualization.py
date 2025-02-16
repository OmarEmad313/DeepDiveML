from manim import *
import numpy as np

class LinearRegressionScene(Scene):
    def construct(self):
        # ---------------------------------------------------
        # 1) Create Axes and a small dataset
        # ---------------------------------------------------
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 12, 2],
            x_length=6,
            y_length=4,
            axis_config={"include_numbers": True},
            tips=False
        )
        axes.to_edge(LEFT)

        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")

        # Suppose the true relationship is y = 1.5x + 2 + noise
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = 1.5 * x_data + 2 + np.random.normal(0, 0.5, len(x_data))

        # Create dots for each data point
        dots = VGroup()
        for (xv, yv) in zip(x_data, y_data):
            dot = Dot(axes.coords_to_point(xv, yv), color=YELLOW)
            dots.add(dot)

        # ---------------------------------------------------
        # 2) Define regression parameters & helper functions
        # ---------------------------------------------------
        theta0 = 0.0  # intercept
        theta1 = 0.0  # slope
        alpha = 0.05  # learning rate
        num_iterations = 6  # number of steps to animate

        def compute_mse(t0, t1, x, y):
            preds = t1 * x + t0
            return np.mean((preds - y)**2)

        def compute_r2(t0, t1, x, y):
            preds = t1 * x + t0
            ss_res = np.sum((preds - y)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            return 1 - (ss_res / ss_tot)

        def gradient_step(t0, t1):
            # Single step of gradient descent
            m = len(x_data)
            preds = t1 * x_data + t0
            errors = preds - y_data
            dtheta1 = (1/m) * np.sum(errors * x_data)
            dtheta0 = (1/m) * np.sum(errors)
            t1_new = t1 - alpha * dtheta1
            t0_new = t0 - alpha * dtheta0
            return t0_new, t1_new

        def get_regression_line(t0, t1):
            x_min, x_max = 0, 6
            start_point = axes.coords_to_point(x_min, t1*x_min + t0)
            end_point   = axes.coords_to_point(x_max, t1*x_max + t0)
            return Line(start_point, end_point, color=RED, stroke_width=4)

        # A function to create a text block with iteration, eq, MSE, R^2
        def get_info_text(iteration, t0, t1, x, y):
            mse_val = compute_mse(t0, t1, x, y)
            r2_val  = compute_r2(t0, t1, x, y)
            eq_str  = f"y = {t1:.2f}x + {t0:.2f}"
            info_str = (
                f"Iteration: {iteration}\n"
                f"{eq_str}\n"
                f"MSE: {mse_val:.2f}\n"
                f"R^2: {r2_val:.2f}"
            )
            text_mob = Text(info_str, font_size=24)
            text_mob.next_to(axes, UR).shift(RIGHT*0.5)
            return text_mob

        # ---------------------------------------------------
        # 3) Add initial objects to the Scene
        # ---------------------------------------------------
        line = get_regression_line(theta0, theta1)
        info_text = get_info_text(0, theta0, theta1, x_data, y_data)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))
        self.play(FadeIn(dots))
        self.play(Create(line), FadeIn(info_text))

        old_info_text = info_text

        # ---------------------------------------------------
        # 4) Animate gradient descent steps
        # ---------------------------------------------------
        for i in range(1, num_iterations+1):
            # 1) Update parameters
            new_t0, new_t1 = gradient_step(theta0, theta1)
            new_line = get_regression_line(new_t0, new_t1)

            # 2) Build new text info
            new_info_text = get_info_text(i, new_t0, new_t1, x_data, y_data)

            # 3) Animate line transform + text transform
            #    We'll do a "ReplacementTransform" so the old text morphs into the new text.
            self.play(
                Transform(line, new_line),
                ReplacementTransform(old_info_text, new_info_text),
                run_time=2
            )

            # 4) Update local variables
            theta0, theta1 = new_t0, new_t1
            old_info_text = new_info_text

        self.wait(2)

# from manim import *
# import numpy as np

# class LinearRegressionScene(Scene):
#     def construct(self):
#         # 1) Create Axes
#         #    - We'll set an x-range from 0 to 6 and a y-range from 0 to 12
#         axes = Axes(
#             x_range=[0, 6, 1],
#             y_range=[0, 12, 2],
#             x_length=6,
#             y_length=4,
#             axis_config={"include_numbers": True},
#             tips=False
#         )
#         axes.to_edge(LEFT)  # shift axes to the left side

#         # Add labels for axes
#         x_label = axes.get_x_axis_label("x")
#         y_label = axes.get_y_axis_label("y")
        
#         # 2) Create a small dataset
#         #    Suppose the true relationship is y = 1.5x + 2, plus noise
#         x_data = np.array([1, 2, 3, 4, 5])
#         y_data = 1.5 * x_data + 2 + np.random.normal(0, 0.5, len(x_data))

#         # 3) Plot the data points as Dots
#         dots = VGroup()
#         for x_val, y_val in zip(x_data, y_data):
#             dot = Dot(axes.coords_to_point(x_val, y_val), color=YELLOW)
#             dots.add(dot)

#         # 4) Define the linear regression parameters
#         theta0 = 0.0  # intercept
#         theta1 = 0.0  # slope
#         alpha = 0.05  # learning rate
#         num_iterations = 6  # how many steps to animate

#         # 5) Define cost and gradient step
#         def compute_cost(t0, t1):
#             # Mean Squared Error (no 1/(2m) factor for simplicity)
#             preds = t1 * x_data + t0
#             return np.mean((preds - y_data) ** 2)

#         def gradient_step(t0, t1):
#             # Single step of gradient descent
#             m = len(x_data)
#             preds = t1 * x_data + t0
#             errors = preds - y_data
#             dtheta1 = (1/m) * np.sum(errors * x_data)
#             dtheta0 = (1/m) * np.sum(errors)
#             t1_new = t1 - alpha * dtheta1
#             t0_new = t0 - alpha * dtheta0
#             return t0_new, t1_new

#         # 6) Helper function to create a Line in Manim for the regression
#         def get_regression_line(t0, t1):
#             # We'll draw from x=0 to x=6
#             x_min, x_max = 0, 6
#             start_point = axes.coords_to_point(x_min, t1*x_min + t0)
#             end_point   = axes.coords_to_point(x_max, t1*x_max + t0)
#             return Line(start_point, end_point, color=RED, stroke_width=4)

#         # 7) Initialize the line and cost text
#         line = get_regression_line(theta0, theta1)
#         cost_value = compute_cost(theta0, theta1)
#         cost_text = Text(f"Cost: {cost_value:.2f}", font_size=24)
#         cost_text.next_to(axes, UR).shift(RIGHT*0.5)

#         # 8) Add everything to the scene
#         self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))
#         self.play(FadeIn(dots))
#         self.play(Create(line), FadeIn(cost_text))

#         # 9) Animate gradient descent steps
#         for i in range(num_iterations):
#             # 1. Compute new params
#             new_t0, new_t1 = gradient_step(theta0, theta1)
#             new_line = get_regression_line(new_t0, new_t1)

#             # 2. Compute new cost
#             new_cost = compute_cost(new_t0, new_t1)
#             new_cost_text = Text(f"Cost: {new_cost:.2f}", font_size=24)
#             new_cost_text.next_to(axes, UR).shift(RIGHT*0.5)

#             # 3. Animate line transformation + cost text
#             #    We'll replace the old line and cost text
#             self.play(
#                 Transform(line, new_line),
#                 Transform(cost_text, new_cost_text),
#                 run_time=2
#             )

#             # Update parameters for next iteration
#             theta0, theta1 = new_t0, new_t1

#         # 10) Pause briefly at the end
#         self.wait(2)



