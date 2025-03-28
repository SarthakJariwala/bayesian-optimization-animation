import math

import numpy as np
# ruff: noqa: F403, F405
from manim import *


class BayesianOptimizationAnimation(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set random seed for reproducibility
        np.random.seed(42)

        # More complex target function (combination of sine waves and quadratic term)
        self.target_func = (
            lambda x: 0.5 * np.sin(3 * x)
            - 0.2 * np.sin(5 * x)
            + 0.7 * np.sin(2 * x)
            + 0.1 * (x - 3) ** 2
            + np.random.normal(0, 0.1)
        )

        # Domain
        self.x_min, self.x_max = 0, 2 * np.pi

        # Kernel parameters for Gaussian Process
        self.length_scale = 0.3  # Reduced to capture more complex function
        self.signal_variance = 1.0

        # Initial samples (1D array for simplicity)
        self.X = np.linspace(self.x_min, self.x_max, 5)
        self.Y = np.array([self.target_func(x) for x in self.X])

        # Standard font size for consistency
        self.label_font_size = 20
        self.small_label_font_size = 16
        self.font = "Monaspace Neon"

        # Animation timing constants
        self.scene_pause = 3  # 3 seconds between scenes
        self.transition_duration = 1.0  # Duration for transitions

        # Colors for different elements
        self.true_func_color = BLUE
        self.sample_point_color = RED
        self.gp_mean_color = PURPLE_B
        self.uncertainty_color = TEAL_C
        self.acquisition_color = YELLOW

        # The true function without noise for visualization
        self.true_func_no_noise = (
            lambda x: 0.5 * np.sin(3 * x)
            - 0.2 * np.sin(5 * x)
            + 0.7 * np.sin(2 * x)
            + 0.1 * (x - 3) ** 2
        )

    def rbf_kernel(self, X1, X2):
        """Radial Basis Function (Squared Exponential) Kernel"""
        # Ensure X1 and X2 are 1D arrays
        X1 = np.atleast_1d(X1)
        X2 = np.atleast_1d(X2)

        # Compute squared distances
        X1_sq = X1[:, np.newaxis] ** 2
        X2_sq = X2**2

        # Compute cross terms
        cross_terms = np.dot(X1[:, np.newaxis], X2[np.newaxis, :])

        # Compute squared distance matrix
        sq_dist = X1_sq + X2_sq.T - 2 * cross_terms

        # Compute kernel
        return self.signal_variance * np.exp(-0.5 * sq_dist / (self.length_scale**2))

    def compute_gp_posterior(self, X_train, Y_train, X_test):
        """Compute Gaussian Process Posterior"""
        # Ensure X_test is a 1D array
        X_test = np.atleast_1d(X_test)

        # Kernel matrices
        K = self.rbf_kernel(X_train, X_train)
        K_star = self.rbf_kernel(X_test, X_train)
        K_star_star = self.rbf_kernel(X_test, X_test)

        # Add small noise to prevent numerical instability
        noise_var = 1e-8
        K += np.eye(len(K)) * noise_var

        # Compute posterior
        K_inv = np.linalg.inv(K)
        mu = K_star @ K_inv @ Y_train
        cov = K_star_star - K_star @ K_inv @ K_star.T

        return mu, np.diag(cov)

    def acquisition_function(self, X_test, X_train, Y_train):
        """Upper Confidence Bound Acquisition Function"""
        mu, sigma = self.compute_gp_posterior(X_train, Y_train, X_test)

        # Balance exploration and exploitation
        kappa = 2.0
        ucb = mu + kappa * np.sqrt(sigma)
        return ucb

    def create_confidence_interval(self, axes, X_test, mu, sigma):
        """Create and return a confidence interval polygon"""
        upper_bound = mu + np.sqrt(sigma)
        lower_bound = mu - np.sqrt(sigma)

        # Combine x and bounds for polygon
        x_poly = np.concatenate([X_test, X_test[::-1]])
        y_poly = np.concatenate([upper_bound, lower_bound[::-1]])

        # Convert to Manim points
        poly_points = [axes.c2p(x, y) for x, y in zip(x_poly, y_poly)]

        # Create polygon for confidence interval
        return Polygon(
            *poly_points, color=self.uncertainty_color, fill_opacity=0.2, stroke_width=0
        )

    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[self.x_min, self.x_max, math.pi / 2],
            y_range=[-2, 2, 0.5],
            x_length=8,
            y_length=5,
        )
        self.add(axes)

        # Title
        title = Text(
            "Bayesian Optimization Process", font_size=24, font=self.font
        ).to_edge(UP)
        self.add(title)

        # INITIAL SCENE: SEQUENTIAL DISPLAY
        # Step 1: Show true function first
        initial_text = Text(
            "True Objective Function", font_size=self.label_font_size, font=self.font
        ).to_edge(DOWN)
        self.add(initial_text)

        true_func_plot = axes.plot(
            self.true_func_no_noise, color=self.true_func_color, stroke_width=2
        )
        true_func_label = (
            Text(
                "True Function",
                font_size=self.label_font_size,
                color=self.true_func_color,
                font=self.font,
            )
            .next_to(axes, UP)
            .shift(LEFT * 2)
        )

        self.play(
            Create(true_func_plot, run_time=self.transition_duration),
            FadeIn(true_func_label, run_time=self.transition_duration),
        )
        self.wait(self.scene_pause)

        # Step 2: Show initial samples
        self.remove(initial_text)
        initial_text = Text(
            "Collecting Initial Samples", font_size=self.label_font_size, font=self.font
        ).to_edge(DOWN)
        self.add(initial_text)

        # Initial sample points with animation
        initial_points = VGroup()
        for x, y in zip(self.X, self.Y):
            point = Dot(axes.c2p(x, y), color=self.sample_point_color)
            initial_points.add(point)

        initial_sample_label = (
            Text(
                "Initial Samples",
                font_size=self.label_font_size,
                color=self.sample_point_color,
                font=self.font,
            )
            .next_to(axes, DOWN)
            .shift(RIGHT * 1)
        )

        # Animate points appearing one by one with a slower transition
        self.play(
            LaggedStart(
                *[
                    GrowFromCenter(point, run_time=self.transition_duration)
                    for point in initial_points
                ],
                lag_ratio=0.4,
            ),
            FadeIn(initial_sample_label, run_time=self.transition_duration),
        )
        self.wait(self.scene_pause)

        # Step 3: Show GP mean
        self.remove(initial_text)
        initial_text = Text(
            "Fitting Gaussian Process", font_size=self.label_font_size, font=self.font
        ).to_edge(DOWN)
        self.add(initial_text)

        # Compute test points
        X_test = np.linspace(self.x_min, self.x_max, 100)

        # Compute initial GP posterior
        mu, sigma = self.compute_gp_posterior(self.X, self.Y, X_test)

        # Plot initial GP mean
        initial_gp_mean = axes.plot_line_graph(
            x_values=X_test, y_values=mu, line_color=self.gp_mean_color, stroke_width=2
        )

        initial_gp_mean_label = (
            Text(
                "Gaussian Process Mean",
                font_size=self.label_font_size,
                color=self.gp_mean_color,
                font=self.font,
            )
            .next_to(axes, UP)
            .shift(RIGHT * 2)
        )  # Position to avoid overlap

        self.play(
            Create(initial_gp_mean, run_time=self.transition_duration),
            FadeIn(initial_gp_mean_label, run_time=self.transition_duration),
            FadeOut(initial_sample_label, run_time=self.transition_duration),
        )
        self.wait(self.scene_pause)

        # Step 4: Show uncertainty
        self.remove(initial_text)
        initial_text = Text(
            "Uncertainty Estimation", font_size=self.label_font_size, font=self.font
        ).to_edge(DOWN)
        self.add(initial_text)

        # Create initial confidence interval
        initial_confidence_interval = self.create_confidence_interval(
            axes, X_test, mu, sigma
        )

        uncertainty_label = (
            Text(
                "Uncertainty",
                font_size=self.label_font_size,
                color=self.uncertainty_color,
                font=self.font,
            )
            .next_to(axes, RIGHT)
            .shift(UP * 1)
        )

        self.play(
            FadeIn(initial_confidence_interval, run_time=self.transition_duration),
            FadeIn(uncertainty_label, run_time=self.transition_duration),
        )
        self.wait(self.scene_pause)

        # Start optimization process
        self.remove(initial_text)

        # Perform Bayesian Optimization iterations
        for iteration in range(5):
            # Update iteration text
            iteration_text = Text(
                f"Iteration {iteration + 1}",
                font_size=self.label_font_size,
                font=self.font,
            ).to_edge(DOWN)
            self.add(iteration_text)

            # Compute test points (already computed for first iteration)
            if iteration > 0:
                X_test = np.linspace(self.x_min, self.x_max, 100)
                mu, sigma = self.compute_gp_posterior(self.X, self.Y, X_test)

                # Update GP mean and confidence interval
                gp_mean = axes.plot_line_graph(
                    x_values=X_test,
                    y_values=mu,
                    line_color=self.gp_mean_color,
                    stroke_width=2,
                )
                confidence_interval = self.create_confidence_interval(
                    axes, X_test, mu, sigma
                )
                gp_mean_label = (
                    Text(
                        "Gaussian Process Mean",
                        font_size=self.label_font_size,
                        color=self.gp_mean_color,
                        font=self.font,
                    )
                    .next_to(axes, UP)
                    .shift(RIGHT * 2)
                )  # Position to avoid overlap

                self.play(
                    FadeOut(initial_gp_mean, run_time=self.transition_duration / 2),
                    FadeOut(
                        initial_confidence_interval,
                        run_time=self.transition_duration / 2,
                    ),
                    FadeOut(
                        initial_gp_mean_label, run_time=self.transition_duration / 2
                    ),
                    FadeIn(gp_mean, run_time=self.transition_duration),
                    FadeIn(confidence_interval, run_time=self.transition_duration),
                    FadeIn(gp_mean_label, run_time=self.transition_duration),
                )

                initial_gp_mean = gp_mean
                initial_confidence_interval = confidence_interval
                initial_gp_mean_label = gp_mean_label

            # Compute acquisition function
            acq_values = self.acquisition_function(X_test, self.X, self.Y)

            # Plot acquisition function in a smaller subplot
            acq_text = (
                Text(
                    "Acquisition Function (UCB)",
                    font_size=self.small_label_font_size,
                    color=self.acquisition_color,
                    font=self.font,
                )
                .to_edge(RIGHT)
                .shift(UP * 2)
            )
            self.play(FadeIn(acq_text, run_time=self.transition_duration))

            # Find next sample point (max of acquisition function)
            next_x_idx = np.argmax(acq_values)
            next_x = X_test[next_x_idx]

            # Highlight the maximum point with a vertical line to x-axis
            x_line = DashedLine(
                start=axes.c2p(next_x, -2),
                end=axes.c2p(next_x, mu[next_x_idx] + np.sqrt(sigma[next_x_idx])),
                color=self.acquisition_color,
                stroke_width=2,
            )
            x_marker = Dot(axes.c2p(next_x, -2), color=self.acquisition_color)
            proposed_point_text = Text(
                "Proposed Point",
                font_size=self.small_label_font_size,
                color=self.acquisition_color,
                font=self.font,
            ).next_to(x_marker, DOWN)

            # Animate the vertical line and proposed point
            self.play(
                Create(x_line, run_time=self.transition_duration),
                GrowFromCenter(x_marker, run_time=self.transition_duration),
                FadeIn(proposed_point_text, run_time=self.transition_duration),
            )

            # Pause to show the proposed point
            self.wait(self.scene_pause)

            # Sample the true function at the proposed point
            next_y = self.target_func(next_x)

            # Create horizontal line to the y-axis for the result
            y_line = DashedLine(
                start=axes.c2p(0, next_y),
                end=axes.c2p(next_x, next_y),
                color=self.sample_point_color,
                stroke_width=2,
            )
            y_marker = Dot(axes.c2p(0, next_y), color=self.sample_point_color)
            result_text = Text(
                "Observed Value",
                font_size=self.small_label_font_size,
                color=self.sample_point_color,
                font=self.font,
            ).next_to(y_marker, LEFT)

            # Add new sample point
            next_point = Dot(axes.c2p(next_x, next_y), color=self.sample_point_color)

            # Show the measurement process with animation
            self.play(
                Create(y_line, run_time=self.transition_duration),
                GrowFromCenter(y_marker, run_time=self.transition_duration),
                FadeIn(result_text, run_time=self.transition_duration),
                GrowFromCenter(next_point, run_time=self.transition_duration),
            )
            self.wait(self.scene_pause)

            # Update training data
            self.X = np.append(self.X, next_x)
            self.Y = np.append(self.Y, next_y)

            # Show uncertainty shrinking by animating the transition
            new_mu, new_sigma = self.compute_gp_posterior(self.X, self.Y, X_test)
            new_confidence_interval = self.create_confidence_interval(
                axes, X_test, new_mu, new_sigma
            )
            new_gp_mean = axes.plot_line_graph(
                x_values=X_test,
                y_values=new_mu,
                line_color=self.gp_mean_color,
                stroke_width=2,
            )

            # Highlight the region where uncertainty is reduced
            highlight_region = self.create_confidence_interval(
                axes,
                X_test[max(0, next_x_idx - 10) : min(len(X_test), next_x_idx + 10)],
                new_mu[max(0, next_x_idx - 10) : min(len(X_test), next_x_idx + 10)],
                new_sigma[max(0, next_x_idx - 10) : min(len(X_test), next_x_idx + 10)],
            )
            highlight_region.set_color(self.acquisition_color).set_opacity(0.3)
            uncertainty_reduced_text = Text(
                "Uncertainty Reduced",
                font_size=self.small_label_font_size,
                font=self.font,
                color=self.acquisition_color,
            ).next_to(highlight_region, UP)

            # Animate the transition to updated GP
            self.play(
                Transform(
                    initial_gp_mean, new_gp_mean, run_time=self.transition_duration
                ),
                Transform(
                    initial_confidence_interval,
                    new_confidence_interval,
                    run_time=self.transition_duration,
                ),
                FadeIn(highlight_region, run_time=self.transition_duration),
                FadeIn(uncertainty_reduced_text, run_time=self.transition_duration),
            )
            self.wait(self.scene_pause)

            # Remove previous iteration-specific objects
            self.play(
                FadeOut(x_line, run_time=self.transition_duration / 2),
                FadeOut(x_marker, run_time=self.transition_duration / 2),
                FadeOut(proposed_point_text, run_time=self.transition_duration / 2),
                FadeOut(y_line, run_time=self.transition_duration / 2),
                FadeOut(y_marker, run_time=self.transition_duration / 2),
                FadeOut(result_text, run_time=self.transition_duration / 2),
                FadeOut(acq_text, run_time=self.transition_duration / 2),
                FadeOut(highlight_region, run_time=self.transition_duration / 2),
                FadeOut(
                    uncertainty_reduced_text, run_time=self.transition_duration / 2
                ),
                FadeOut(iteration_text, run_time=self.transition_duration / 2),
            )

            # We keep the sample points visible throughout the animation

            # Update for next iteration
            initial_gp_mean_label = (
                Text(
                    "Gaussian Process Mean",
                    font_size=self.label_font_size,
                    color=self.gp_mean_color,
                    font=self.font,
                )
                .next_to(axes, UP)
                .shift(RIGHT * 2)
            )  # Position to avoid overlap
            self.play(
                FadeIn(initial_gp_mean_label, run_time=self.transition_duration / 2)
            )

            # Wait between iterations
            self.wait(self.scene_pause)


# Render the animation
if __name__ == "__main__":
    from manim import config

    # Updated configuration method
    config.video_dir = "./videos"
    config.media_dir = "./media"
    config.pixel_height = 1080
    config.pixel_width = 1920

    # Render the scene
    scene = BayesianOptimizationAnimation()
    scene.render()
