import svgwrite

# Helper methods
def _break_lines(text, limit=50):
    lines = []
    curr_line = ""
    for token in text.split(" "):
        if len(curr_line) + 1 + len(token) <= limit:
            curr_line += token + " "
        else:
            lines.append(curr_line[:-1])
            curr_line = token + " "
    lines.append(curr_line[:-1])
    return lines


def _get_rect_style(color, opacity):
    return """opacity:%0.5f;
            fill:%s;
            fill-opacity:1;
            stroke:none;
            stroke-width:0.26499999;
            stroke-linecap:round;
            stroke-linejoin:miter;
            stroke-miterlimit:4;
            stroke-dasharray:none;
            stroke-dashoffset:0;
            stroke-opacity:1""" % (
        opacity,
        color,
    )


def _get_text_style(font_size):
    return f"""font-style: normal;
            font-variant: normal;
            font-weight: normal;
            font-stretch: normal;
            font-size: {font_size:0.2f}px;
            line-height: 125%;
            font-family: "Courier";
            -inkscape-font-specification: "Courier";
            font-variant-ligatures: none;
            font-variant-caps: normal;
            font-variant-numeric: normal;
            text-align: start;
            writing-mode: lr-tb;
            text-anchor: start;
            stroke-width: 0.26458332px"""


MARGIN = 10


def visualize_activations(
    tokens,
    activations,
    darken=2,
    colors=["#d35f5f", "#00aad4"],
    text_direction="ltr",
    char_limit=60,
    font_size=20,
):
    """
    Visualize activation values for a particular neuron on some text.

    This method returns an SVG drawing of text with every token's background
    color set according to the passed in activation values (red for negative
    values and blue for positive).

    Parameters
    ----------
    tokens : list of str
        List of tokens over which the activations have been computed. In the
        rendered image, tokens will be separated by a single space.
    activations: list of float
        List of activation values, one per token.
    darken : int, optional
        Number of times to render the red/blue background. Increasing this
        value will reduce contrast but may help in better distinguishing between
        tokens. Defaults to 2
    colors : list of str, optional
        List of two elements, the first indicating the color of the lowest
        activation value and the second indicating the color of the highest
        activation value. Defaults to shades of red and blue respectively
    text_direction : str, optional
        One of ``ltr`` or ``rtl``, indicating if the language being rendered is
        written left to right or right to left. Defaults to ``ltr``
    char_limit : int, optional
        Maximum number of characters per line. Defaults to 60
    font_size : int, optional
        Font size in pixels. Defaults to 20px

    Returns
    -------
    rendered_svg : svgwrite.Drawing
        A SVG object that you can either save to file, convert into a png within
        python using an external library like Pycairo, or display in a notebook
        using the ``display`` from the module ``IPython.display``
    """
    ################################ Validation ################################
    valid_text_directions = ["ltr", "rtl"]
    text_direction = text_direction.lower()
    assert (
        text_direction in valid_text_directions
    ), f"text_direction must be one of {valid_text_directions}"

    assert len(tokens) == len(
        activations
    ), f"Number of tokens and activations must match"

    ############################## Drawing Setup ###############################
    text = " ".join(tokens)

    # Estimate individual character sizes
    char_width = font_size * 0.601  # Magic number for Courier font
    char_height = font_size * 1.25  # 1.25 is line height of rendered font

    # Compute number of lines
    lines = _break_lines(text, limit=char_limit)

    # Compute image size based on character sizes and number of lines
    image_height = len(lines) * char_height * 1.2
    image_width = (char_limit + 1) * char_width

    # Create drawing canvas
    dwg = svgwrite.Drawing("tmp.svg", size=(image_width, image_height), profile="full")
    dwg.viewbox(0, 0, image_width, image_height)
    group = dwg.g()

    ####################### Activation Rendering limits ########################
    scores = activations
    max_score = max(scores)
    min_score = abs(min(scores))
    limit = max(max_score, min_score)

    for _ in range(darken):
        word_idx = 0
        line_horizontal_offsets = []
        for line_idx, line in enumerate(lines):
            char_idx = 0
            words = line.split(" ")
            if text_direction == "rtl":
                words = reversed(words)
            for word in words:
                score = scores[word_idx]
                if score > 0:
                    color = colors[1]
                    opacity = score / limit
                else:
                    color = colors[0]
                    opacity = abs(score) / limit

                # Add rectangle for every character in current word
                for _ in word:
                    rect_position = (char_idx * char_width, 7 + line_idx * char_height)
                    rect_size = (f"{char_width:0.3f}px", f"{char_height:0.3f}px")
                    group.add(
                        dwg.rect(
                            insert=rect_position,
                            size=rect_size,
                            style=_get_rect_style(color, opacity),
                        )
                    )
                    char_idx += 1

                # Add rectangle for empty space after word
                final_rect_pos = (char_idx * char_width, 7 + line_idx * char_height)
                final_rect_size = (f"{char_width:0.3f}px", f"{char_height:0.3f}px")
                group.add(
                    dwg.rect(
                        insert=final_rect_pos,
                        size=final_rect_size,
                        style=_get_rect_style(color, opacity),
                    )
                )

                char_idx += 1
                word_idx += 1
            if text_direction == "ltr":
                line_horizontal_offsets.append(0)
            else:
                line_horizontal_offsets.append(char_idx * char_width)

        # Draw the actual text over the drawn rectangles
        for line_idx, line in enumerate(lines):
            text_insert = (
                line_horizontal_offsets[line_idx],
                font_size * 1.25 * (line_idx + 1),
            )
            text = dwg.text(
                line, insert=text_insert, fill="black", style=_get_text_style(font_size)
            )
            group.add(text)

    dwg.add(group)

    return dwg
