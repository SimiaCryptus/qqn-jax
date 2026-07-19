clean up "examples"
clean up companion md files
add objective evaluation counts (fwd and backward) to the evaluation section
    "estimated function/grad evals" is always null in reports


add a "null" line search that always accepts the t=1 oracle point; if no oracle point, a configurable scaling of the gradient
add a linear interpolation mode that simply interpolates between the origin and the oracle point - this is the opposite of the spline variation, and throws out gradient (unless there is no oracle point)
