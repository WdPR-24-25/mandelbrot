package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"runtime"
	"sync"
	"time"
)

func HSBToRGB(h, s, v float64) (uint8, uint8, uint8) {
	if s == 0 {
		rgb := uint8(v * 255)
		return rgb, rgb, rgb
	}
	h = h * 6
	if h == 6 {
		h = 0
	}
	i := math.Floor(h)
	v1 := v * (1 - s)
	v2 := v * (1 - s*(h-i))
	v3 := v * (1 - s*(1-(h-i)))
	var r, g, b float64
	switch int(i) {
	case 0:
		r, g, b = v, v3, v1
	case 1:
		r, g, b = v2, v, v1
	case 2:
		r, g, b = v1, v, v3
	case 3:
		r, g, b = v1, v2, v
	case 4:
		r, g, b = v3, v1, v
	default:
		r, g, b = v, v1, v2
	}
	return uint8(r * 255), uint8(g * 255), uint8(b * 255)
}

type Mandelbrot struct {
	minRe   float64
	maxRe   float64
	minIm   float64
	maxIm   float64
	maxIter int
	wg      *sync.WaitGroup
}

type Block struct {
	x      int
	y      int
	width  int
	height int
}

func (mb *Mandelbrot) calculate(x, y float64) color.Color {
	c := complex(x, y)
	z := complex(0, 0)

	for i := 0; i < mb.maxIter; i++ {
		if real(z)*real(z)+imag(z)*imag(z) > 4 {
			hue := float64(i) / float64(mb.maxIter) * 2.5
			r, g, b := HSBToRGB(hue, 1.0, 1.0)
			return color.RGBA{r, g, b, 255}
		}

		z = z*z + c
	}

	return color.Black
}

func (mb *Mandelbrot) render(img *image.RGBA, block Block) {
	if mb.wg != nil {
		defer mb.wg.Done()
	}

	dx := (mb.maxRe - mb.minRe) / float64(img.Rect.Dx())
	dy := (mb.maxIm - mb.minIm) / float64(img.Rect.Dy())

	for py := block.y; py < block.y+block.height; py++ {
		y := mb.minIm + float64(py)*dy
		for px := block.x; px < block.x+block.width; px++ {
			x := mb.minRe + float64(px)*dx
			pColor := mb.calculate(x, y)
			img.Set(px, py, pColor)
		}
	}
}

func mandelbrotParallel(mandelbrot *Mandelbrot, img *image.RGBA) {
	cores := runtime.NumCPU()
	size := img.Bounds().Size().X
	gridSize := int(math.Sqrt(float64(cores)))
	blockSize := size / gridSize

	var wg sync.WaitGroup
	mandelbrot.wg = &wg

	for y := 0; y < size; y += blockSize {
		for x := 0; x < size; x += blockSize {
			width := blockSize
			height := blockSize

			if x+width > size {
				width = size - x
			}
			if y+height > size {
				height = size - y
			}

			wg.Add(1)
			go mandelbrot.render(img, Block{x, y, width, height})
		}
	}

	wg.Wait()
}

const (
	SERIAL   = "serial"
	PARALLEL = "parallel"
)

func main() {
	filepath := flag.String("outfile", "", "file to save output to")
	mode := flag.String("mode", SERIAL, "mode of execution")
	saveImage := flag.Bool("save-image", false, "mode of execution")
	flag.Parse()

	var outfile *os.File

	if *filepath == "" {
		outfile = os.Stdout
	} else {
		var err error
		outfile, err = os.Create(*filepath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating file %s: %v\n", *filepath, err)
			os.Exit(1)
		}
		defer outfile.Close()
	}

	mandelbrot := Mandelbrot{
		minRe:   -2.0,
		maxRe:   1.0,
		minIm:   -1.5,
		maxIm:   1.5,
		maxIter: 200,
	}

	imageSaved := false

	for size := 16; size < 1<<14; size *= 2 {
		img := image.NewRGBA(image.Rect(0, 0, size, size))
		start := time.Now()

		repeat := 10.0
		for i := 0.0; i < repeat; i++ {
			switch *mode {
			case SERIAL:
				mandelbrot.render(img, Block{0, 0, size, size})
			case PARALLEL:
				mandelbrotParallel(&mandelbrot, img)
			}
		}

		duration := float64(time.Since(start).Microseconds()) / repeat
		fmt.Fprintln(outfile, size, duration)

		if *saveImage && !imageSaved {
			imageSaved = true
			filename := fmt.Sprintf("mandelbrot-%d.png", size)
			file, _ := os.Create(filename)
			defer file.Close()
			png.Encode(file, img)
		}
	}
}
