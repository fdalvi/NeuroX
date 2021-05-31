import svgwrite

def break_lines(text, limit=50):
    lines = []
    curr_line = ""
    for token in text.split(' '):
        if len(curr_line) + 1 + len(token) < limit:
            curr_line += token + " "
        else:
            lines.append(curr_line[:-1])
            curr_line = token + " "
    lines.append(curr_line[:-1])
    return lines

def get_rect_style(color, opacity):
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
            stroke-opacity:1""" % (opacity, color)

def get_text_style(font_size):
    return """font-style:normal;
            font-variant:normal;
            font-weight:normal;
            font-stretch:normal;
            font-size:%0.2fpx;
            line-height:125%%;
            font-family:monospace;
            -inkscape-font-specification:'Consolas, Normal';
            font-variant-ligatures:none;
            font-variant-caps:normal;
            font-variant-numeric:normal;
            text-align:start;
            writing-mode:lr-tb;
            text-anchor:start;
            stroke-width:0.26458332px""" % (font_size)

FONT_SIZE = 20
MARGIN = 10
CHAR_LIMIT = 61

def visualize_activations(text, activations, darken=2, colors=["#d35f5f", "#00aad4"]):
    char_width = FONT_SIZE*0.59
    char_height = FONT_SIZE*1.25

    lines = break_lines(text, limit=CHAR_LIMIT)
    scores = activations

    image_height = len(lines) * char_height * 1.2
    image_width = CHAR_LIMIT * char_width

    dwg = svgwrite.Drawing("tmp.svg", size=(image_width, image_height),
                        profile='full')
    dwg.viewbox(0, 0, image_width, image_height)

    offset = 0

    group = dwg.g()
    for _ in range(darken):
        word_idx = 0
        for line_idx, line in enumerate(lines):
            char_idx = 0
            max_score = max(scores)
            min_score = abs(min(scores))
            limit = max(max_score, min_score)
            for word in line.split(' '):
                score = scores[word_idx]
                if score > 0:
                    color = colors[1]
                    opacity = score/limit
                else:
                    color = colors[0]
                    opacity = abs(score)/limit

                for _ in word:
                    rect_insert = (0 + char_idx * char_width, offset + 7 + line_idx * char_height)
                    rect_size = ("%.2fpx"%(char_width), "%0.2fpx"%(char_height))
                    group.add(
                        dwg.rect(insert=rect_insert,
                                size=rect_size,
                                style=get_rect_style(color, opacity)
                                )
                    )
                    char_idx += 1

                final_rect_insert = (0 + char_idx * char_width, offset + 7 + line_idx * char_height)
                final_rect_size = ("%.2fpx"%(char_width), "%0.2fpx"%(char_height))
                group.add(
                    dwg.rect(insert=final_rect_insert,
                            size=final_rect_size,
                            style=get_rect_style(color, opacity)
                            )
                )

                char_idx += 1
                word_idx += 1

        for line_idx, line in enumerate(lines):
            text_insert = (0, offset + FONT_SIZE*1.25*(line_idx+1))
            text = dwg.text(line,
                            insert=text_insert,
                            fill='black',
                            style=get_text_style(FONT_SIZE))
            group.add(text)
    offset += FONT_SIZE*1.25*len(lines) + MARGIN

    dwg.add(group)

    return dwg