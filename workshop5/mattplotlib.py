"""
dead-simple terminal plotting library by matt.
"""

import numpy as np
import einops


# # # 
# PLOT BASE CLASS


class plot:
    """
    Base class representing a 2d character array as a list of lines.
    Provides methods for converting to a string, along with operations
    for horizontal (&) and vertical (^) stacking.
    """
    def __init__(self, height: int, width: int, lines: list[str]):
        self.height = height
        self.width = width
        self.lines = lines

    def __str__(self) -> str:
        return "\n".join(self.lines)

    def __and__(self, other):
        return hstack(self, other)

    def __xor__(self, other):
        return vstack(self, other)


# # # 
# DATA PLOTTING CLASSES


class image(plot):
    """
    Render a small image using a grid of unicode half-characters with
    different foreground and background colours to represent pairs of
    pixels.

    TODO: document input and colormap formats.
    """
    def __init__(self, im, colormap=None):
        # preprocessing: all inputs become float[h, w, rgb] with even h, w
        im = np.asarray(im)
        if len(im.shape) == 2 and colormap is None:
            # greyscale or indexed and no colormap -> uniform colourisation
            im = einops.repeat(im, 'h w -> h w 3')
        elif colormap is not None:
            # indexed, greyscale, or rgb and compatible colormap -> mapped rgb
            im = colormap(im)
        # pad to even height and width (width is not strictly necessary)
        im = np.pad(
            array=im,
            pad_width=(
                (0, im.shape[0] % 2),
                (0, im.shape[1] % 2),
                (0, 0),
            ),
            mode='constant',
            constant_values=0.,
        )

        # processing: stack into fg/bg format
        stacked = einops.rearrange(im, '(h fgbg) w c -> h w fgbg c', fgbg=2)

        # render the image lines as unicode strings with ansi color codes
        lines = [
            "".join([switch_color(fg, bg) + "▀" for fg, bg in row])
            + reset_color()
            for row in stacked
        ]

        # form a plot object
        super().__init__(
            height=stacked.shape[0],
            width=stacked.shape[1],
            lines=lines,
        )

    def __repr__(self):
        return f"image(height={self.height}, width={self.width})"


class scatter(plot):
    """
    Render a scatterplot using a grid of braille unicode characters.
    """
    def __init__(
        self,
        data: np.ndarray, # float[n, 2]
        height: int = 10,
        width: int = 30,
        yrange: tuple[float, float] | None = None,
        xrange: tuple[float, float] | None = None,
        color: np.ndarray | None = None,            # float[3] (rgb 0 to 1)
        check_bounds: bool = False,
    ):
        # todo: check shape
        data = np.asarray(data)
        color = np.asarray(color)
        
        # determine data bounds
        xmin, ymin = data.min(axis=0)
        xmax, ymax = data.max(axis=0)
        if xrange is None:
            xrange = (xmin, xmax)
        else:
            xmin, xmax = xrange
        if yrange is None:
            yrange = (ymin, ymax)
        else:
            ymin, ymax = yrange
        # optional check
        if check_bounds:
            out_x = xmin < xrange[0] or xmax > xrange[1]
            out_y = ymin < yrange[0] or ymax > yrange[1]
            if out_x or out_y:
                raise ValueError("Scatter points out of range")
        
        # quantise 2d float coordinates to data grid
        dots, *_bins = np.histogram2d(
            x=data[:,0],
            y=data[:,1],
            bins=(2*width, 4*height),
            range=(xrange, yrange),
        )
        dots = dots.T     # we want y first
        dots = dots[::-1] # correct y for top-down drawing
        
        # render data grid as a grid of braille characters
        grid = [[" " for _ in range(width)] for _ in range(height)]
        bgrid = braille_encode(dots > 0)
        for i in range(height):
            for j in range(width):
                if bgrid[i, j]:
                    braille_char = chr(0x2800+bgrid[i, j])
                    grid[i][j] = switch_color(fg=color) + braille_char

        # render braille grid as lines
        lines = ["".join(row)+reset_color() for row in grid]

        super().__init__(
            height=height,
            width=width,
            lines=lines,
        )
        self.xrange = xrange
        self.yrange = yrange
        self.num_points = len(data)

    def __repr__(self):
        return (
            f"scatter(height={self.height}, width={self.width}, "
            f"data=<{self.num_points} points on "
            f"[{self.xrange[0]},{self.xrange[1]}]x"
            f"[{self.yrange[0]},{self.yrange[1]}]>)"
        )


class text(plot):
    """
    One or more lines of ASCII text.
    TODO: allow alignment and resizing.
    TODO: account for non-printable and wide characters.
    """
    def __init__(self, text: str):
        lines = text.splitlines()
        height = len(lines)
        width = max(len(line) for line in lines)
        padded_lines = [line.ljust(width) for line in lines]
        super().__init__(
            height=height,
            width=width,
            lines=padded_lines,
        )

    def __repr__(self):
        if len(self.lines) > 1 or len(self.lines[0]) > 8:
            preview = self.lines[0][:5] + "..."
        else:
            preview = self.lines[0][:8]
        return (
            f"text(height={self.height}, width={self.width}, "
            f"text={preview!r})"
        )


# # # 
# ARRANGEMENT CLASSES


class blank(plot):
    """
    A rectangle of blank space.
    """
    def __init__(height: int, width: int):
        super().__init__(
            height=height,
            width=width,
            lines=[" " * width] * height,
        )

    def __repr__(self):
        return f"blank(height={self.height}, width={self.width})"


class hstack(plot):
    """
    Horizontally arrange a group of plots.
    """
    def __init__(self, *plots):
        height = max(p.height for p in plots)
        width = sum(p.width for p in plots)
        lines = [
            "".join([
                p.lines[i] if i < p.height else p.width * " "
                for p in plots
            ])
            for i in range(height)
        ]
        super().__init__(
            height=height,
            width=width,
            lines=lines,
        )
        self.plots = plots

    def __repr__(self):
        return (
            f"hstack(height={self.height}, width={self.width}, "
            f"plots={self.plots!r})"
        )


class vstack(plot):
    """
    Vertically arrange a group of plots.
    """
    def __init__(self, *plots):
        height = sum(p.height for p in plots)
        width = max(p.width for p in plots)
        lines = [l + " " * (width - p.width) for p in plots for l in p.lines]
        super().__init__(
            height=height,
            width=width,
            lines=lines,
        )
        self.plots = plots

    def __repr__(self):
        return (
            f"vstack(height={self.height}, width={self.width}, "
            f"plots={self.plots!r})"
        )


class wrap(plot):
    """
    Horizontally and vertically arrange a group of plots.
    """
    def __init__(self, *plots, cols=None):
        cell_height = max(p.height for p in plots)
        cell_width = max(p.width for p in plots)
        if cols is None:
            cols = 80 // cell_width
        # wrap list of plots into groups, of length `cols` (except last)
        wrapped_plots = []
        for i, plot in enumerate(plots):
            if i % cols == 0:
                wrapped_plots.append([])
            wrapped_plots[-1].append(plot)
        # combine functionality of hstack/vstack
        lines = [
            "".join([
                p.lines[i] + " " * (cell_width - p.width)
                if i < p.height else " " * cell_width
                for p in group
            ])
            for group in wrapped_plots
            for i in range(cell_height)
        ]
        # done!
        super().__init__(
            height=len(lines),
            width=min(len(plots), cols) * cell_width,
            lines=lines,
        )
        self.wrapped_plots = wrapped_plots

    def __repr__(self):
        return (
            f"wrap(height={self.height}, width={self.width}, "
            f"plots={self.plots!r})"
        )


class border(plot):
    """
    Put a unicode border around a plot.
    """
    class Style:
        LIGHT  = "─│┌┐└┘"
        HEAVY  = "━┃┏┓┗┛"
        DOUBLE = "═║╔╗╚╝"
        BLANK  = "      "
        ROUND  = "─│╭╮╰╯"
        BUMPER = "─│▛▜▙▟"

    def __init__(self, plot: plot, style: Style = Style.ROUND):
        bordered_lines = [
            style[2] + style[0] * plot.width + style[3],
            *[style[1] + line + style[1] for line in plot.lines],
            style[4] + style[0] * plot.width + style[5],
        ]
        super().__init__(
            height=plot.height+2,
            width=plot.width+2,
            lines=bordered_lines,
        )
        self.plot = plot
    
    def __repr__(self):
        style = self.lines[0][0]
        return f"border(style={style!r}, plot={self.plot!r})"


class center(plot):
    """
    Put blank space around a plot.
    """

    def __init__(
        self,
        plot: plot,
        height: int | None = None,
        width: int | None = None,
    ):
        height = plot.height if height is None else max(height, plot.height)
        width = plot.width if width is None else max(width, plot.width)
        def _center(inner_size, outer_size):
            diff = outer_size - inner_size
            left = diff // 2
            right = left + (diff % 2)
            return left, right
        left, right = _center(plot.width, width)
        above, below = _center(plot.height, height)
        centered_lines = (
              [" "*width] * above
            + [" "*left + line + " "*right for line in plot.lines]
            + [" "*width] * below
        )
        super().__init__(
            height=height,
            width=width,
            lines=centered_lines,
        )
        self.plot = plot
    
    def __repr__(self):
        return (
            f"center(height={self.height}, width={self.width}, "
            f"plot={self.plot!r})"
        )


# # # 
# COLORMAPS


def reds(x):
    """
    Red colormap. Simply embeds greyscale value into red channel.
    """
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 0] = x
    return rgb


def greens(x):
    """
    Green colormap. Simply embeds greyscale value into green channel.
    """
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 1] = x
    return rgb


def blues(x):
    """
    Blue colormap. Simply embeds greyscale value into blue channel.
    """
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 2] = x
    return rgb


def yellows(x):
    """
    Yellow colormap. Simply embeds greyscale value into red and green
    channels.
    """
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 0] = x
    rgb[..., 1] = x
    return rgb


def magentas(x):
    """
    Magenta colormap. Simply embeds greyscale value into red and blue
    channels.
    """
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 0] = x
    rgb[..., 2] = x
    return rgb


def cyans(x):
    """
    Cyan colormap. Simply embeds greyscale value into green and blue
    channels.
    """
    rgb = np.zeros((*x.shape, 3))
    rgb[..., 1] = x
    rgb[..., 2] = x
    return rgb


def viridis(x):
    """
    Viridis colormap.

    Details: https://youtu.be/xAoljeRJ3lU
    """
    return np.array([
        [.267,.004,.329],[.268,.009,.335],[.269,.014,.341],[.271,.019,.347],
        [.272,.025,.353],[.273,.031,.358],[.274,.037,.364],[.276,.044,.370],
        [.277,.050,.375],[.277,.056,.381],[.278,.062,.386],[.279,.067,.391],
        [.280,.073,.397],[.280,.078,.402],[.281,.084,.407],[.281,.089,.412],
        [.282,.094,.417],[.282,.100,.422],[.282,.105,.426],[.283,.110,.431],
        [.283,.115,.436],[.283,.120,.440],[.283,.125,.444],[.283,.130,.449],
        [.282,.135,.453],[.282,.140,.457],[.282,.145,.461],[.281,.150,.465],
        [.281,.155,.469],[.280,.160,.472],[.280,.165,.476],[.279,.170,.479],
        [.278,.175,.483],[.278,.180,.486],[.277,.185,.489],[.276,.190,.493],
        [.275,.194,.496],[.274,.199,.498],[.273,.204,.501],[.271,.209,.504],
        [.270,.214,.507],[.269,.218,.509],[.267,.223,.512],[.266,.228,.514],
        [.265,.232,.516],[.263,.237,.518],[.262,.242,.520],[.260,.246,.522],
        [.258,.251,.524],[.257,.256,.526],[.255,.260,.528],[.253,.265,.529],
        [.252,.269,.531],[.250,.274,.533],[.248,.278,.534],[.246,.283,.535],
        [.244,.287,.537],[.243,.292,.538],[.241,.296,.539],[.239,.300,.540],
        [.237,.305,.541],[.235,.309,.542],[.233,.313,.543],[.231,.318,.544],
        [.229,.322,.545],[.227,.326,.546],[.225,.330,.547],[.223,.334,.548],
        [.221,.339,.548],[.220,.343,.549],[.218,.347,.550],[.216,.351,.550],
        [.214,.355,.551],[.212,.359,.551],[.210,.363,.552],[.208,.367,.552],
        [.206,.371,.553],[.204,.375,.553],[.203,.379,.553],[.201,.383,.554],
        [.199,.387,.554],[.197,.391,.554],[.195,.395,.555],[.194,.399,.555],
        [.192,.403,.555],[.190,.407,.556],[.188,.410,.556],[.187,.414,.556],
        [.185,.418,.556],[.183,.422,.556],[.182,.426,.557],[.180,.429,.557],
        [.179,.433,.557],[.177,.437,.557],[.175,.441,.557],[.174,.445,.557],
        [.172,.448,.557],[.171,.452,.557],[.169,.456,.558],[.168,.459,.558],
        [.166,.463,.558],[.165,.467,.558],[.163,.471,.558],[.162,.474,.558],
        [.160,.478,.558],[.159,.482,.558],[.157,.485,.558],[.156,.489,.557],
        [.154,.493,.557],[.153,.497,.557],[.151,.500,.557],[.150,.504,.557],
        [.149,.508,.557],[.147,.511,.557],[.146,.515,.556],[.144,.519,.556],
        [.143,.522,.556],[.141,.526,.555],[.140,.530,.555],[.139,.533,.555],
        [.137,.537,.554],[.136,.541,.554],[.135,.544,.554],[.133,.548,.553],
        [.132,.552,.553],[.131,.555,.552],[.129,.559,.551],[.128,.563,.551],
        [.127,.566,.550],[.126,.570,.549],[.125,.574,.549],[.124,.578,.548],
        [.123,.581,.547],[.122,.585,.546],[.121,.589,.545],[.121,.592,.544],
        [.120,.596,.543],[.120,.600,.542],[.119,.603,.541],[.119,.607,.540],
        [.119,.611,.538],[.119,.614,.537],[.119,.618,.536],[.120,.622,.534],
        [.120,.625,.533],[.121,.629,.531],[.122,.633,.530],[.123,.636,.528],
        [.124,.640,.527],[.126,.644,.525],[.128,.647,.523],[.130,.651,.521],
        [.132,.655,.519],[.134,.658,.517],[.137,.662,.515],[.140,.665,.513],
        [.143,.669,.511],[.146,.673,.508],[.150,.676,.506],[.153,.680,.504],
        [.157,.683,.501],[.162,.687,.499],[.166,.690,.496],[.170,.694,.493],
        [.175,.697,.491],[.180,.701,.488],[.185,.704,.485],[.191,.708,.482],
        [.196,.711,.479],[.202,.715,.476],[.208,.718,.472],[.214,.722,.469],
        [.220,.725,.466],[.226,.728,.462],[.232,.732,.459],[.239,.735,.455],
        [.246,.738,.452],[.252,.742,.448],[.259,.745,.444],[.266,.748,.440],
        [.274,.751,.436],[.281,.755,.432],[.288,.758,.428],[.296,.761,.424],
        [.304,.764,.419],[.311,.767,.415],[.319,.770,.411],[.327,.773,.406],
        [.335,.777,.402],[.344,.780,.397],[.352,.783,.392],[.360,.785,.387],
        [.369,.788,.382],[.377,.791,.377],[.386,.794,.372],[.395,.797,.367],
        [.404,.800,.362],[.412,.803,.357],[.421,.805,.351],[.430,.808,.346],
        [.440,.811,.340],[.449,.813,.335],[.458,.816,.329],[.468,.818,.323],
        [.477,.821,.318],[.487,.823,.312],[.496,.826,.306],[.506,.828,.300],
        [.515,.831,.294],[.525,.833,.288],[.535,.835,.281],[.545,.838,.275],
        [.555,.840,.269],[.565,.842,.262],[.575,.844,.256],[.585,.846,.249],
        [.595,.848,.243],[.606,.850,.236],[.616,.852,.230],[.626,.854,.223],
        [.636,.856,.216],[.647,.858,.209],[.657,.860,.203],[.668,.861,.196],
        [.678,.863,.189],[.688,.865,.182],[.699,.867,.175],[.709,.868,.169],
        [.720,.870,.162],[.730,.871,.156],[.741,.873,.149],[.751,.874,.143],
        [.762,.876,.137],[.772,.877,.131],[.783,.879,.125],[.793,.880,.120],
        [.804,.882,.114],[.814,.883,.110],[.824,.884,.106],[.835,.886,.102],
        [.845,.887,.099],[.855,.888,.097],[.866,.889,.095],[.876,.891,.095],
        [.886,.892,.095],[.896,.893,.096],[.906,.894,.098],[.916,.896,.100],
        [.926,.897,.104],[.935,.898,.108],[.945,.899,.112],[.955,.901,.118],
        [.964,.902,.123],[.974,.903,.130],[.983,.904,.136],[.993,.906,.143],
    ])[(np.clip(x, 0., 1.) * (255)).astype(int)]


def sweetie16(x):
    """
    Sweetie-16 colour palette.

    Details: https://lospec.com/palette-list/sweetie-16
    """
    return np.array([
        [.101,.109,.172],[.364,.152,.364],[.694,.243,.325],[.937,.490,.341],
        [.999,.803,.458],[.654,.941,.439],[.219,.717,.392],[.145,.443,.474],
        [.160,.211,.435],[.231,.364,.788],[.254,.650,.964],[.450,.937,.968],
        [.956,.956,.956],[.580,.690,.760],[.337,.423,.525],[.2  ,.235,.341],
    ])[x]


def pico8(x):
    """
    PICO-8 colour palette.

    Details: https://pico-8.fandom.com/wiki/Palette
    """
    return (np.array([
        [  0,   0,   0], [ 29,  43,  83], [126,  37,  83], [  0, 135,  81],
        [171,  82,  54], [ 95,  87,  79], [194, 195, 199], [255, 241, 232],
        [255,   0,  77], [255, 163,   0], [255, 236,  39], [  0, 228,  54],
        [ 41, 173, 255], [131, 118, 156], [255, 119, 168], [255, 204, 170],
    ]) / 255)[x]


# # # 
# UNICODE HELPER FUNCTIONS


def braille_encode(a):
    """
    Turns a HxW array of booleans into a (H//4)x(W//2) array of braille
    binary codes (suitable for specifying unicode codepoints, just add
    0x2800).
    
    braille symbol:                 binary digit representation:
                    0-o o-1
                    2-o o-3   ---->     0 b  0 0  0 0 0  0 0 0
                    4-o o-5                  | |  | | |  | | |
                    6-o o-7                  7 6  5 3 1  4 2 0
    """
    r = einops.rearrange(a, '(h h4) (w w2) -> (h4 w2) h w', h4=4, w2=2)
    b = (
          r[0]      | r[1] << 3 
        | r[2] << 1 | r[3] << 4 
        | r[4] << 2 | r[5] << 5 
        | r[6] << 6 | r[7] << 7
    )
    return b
        

def switch_color(
    fg: np.ndarray | None = None, # float[3] (rgb 0 to 1)
    bg: np.ndarray | None = None, # float[3] (rgb 0 to 1)
) -> str:
    """
    ANSI control code that switches string color into the given fg and bg
    colors. Switch back later with the code reset_color().
    """
    if fg is not None:
        fgr, fgg, fgb = (255 * fg).astype(np.uint8)
        fgcode = f"\033[38;2;{fgr};{fgg};{fgb}m"
    else:
        fgcode = ""
    if bg is not None:
        bgr, bgg, bgb = (255 * bg).astype(np.uint8)
        bgcode = f"\033[48;2;{bgr};{bgg};{bgb}m"
    else:
        bgcode = ""
    return fgcode + bgcode


def reset_color() -> str:
    """
    ANSI control code that resets string to normal colours, fonts etc.
    """
    return "\033[0m"


# # # 
# DEMO / TEST

if __name__ == "__main__":
    size = 14
    u = np.random.rand(size**2).reshape(size,size)
    i = np.eye(size)
    g = np.clip(np.random.normal(size=(size, size)) + 3, 0, 6) / 6
    plot = (
        border(
            center(text("G'day mattplotlib"), height=3, width=46),
            style=border.Style.DOUBLE,
        )
        ^ border(
            text("uniform:")  ^ image(u, colormap=reds),
            style=border.Style.LIGHT,
        )
        & border(
            text("identity:") ^ image(i, colormap=greens),
            style=border.Style.HEAVY,
        )
        & border(
            text("gaussian:") ^ image(g, colormap=blues),
            style=border.Style.DOUBLE,
        )
        ^ border(
            text("uniform:")  ^ image(u, colormap=yellows),
            style=border.Style.ROUND,
        )
        & border(
            text("identity:") ^ image(i, colormap=magentas),
            style=border.Style.BLANK,
        )
        & border(
            text("gaussian:") ^ image(g, colormap=cyans),
            style=border.Style.BUMPER,
        )
        ^ border(scatter(
            data=np.random.normal(size=(300, 2)),
            height=18,
            width=46,
            xrange=(-5, +5),
            yrange=(-4, +4),
            color=(0,1,0)
        ), style=border.Style.ROUND)
    )
    print(repr(plot))
    print(plot)
