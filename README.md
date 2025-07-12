<h1>DTCWT-based Image Fusion</h1>

<div>
In this project I replicated <a href="https://ieeexplore.ieee.org/document/1369329">Callaghan et al., 2004</a>'s process to fuse the intensity and texture gradients of an image using the Dual Tree Complex Wavelet Transform (DTCWT).
</div> </br>
<div align="center">
    <img src="images/lena_large.png" width=256px >
</div>

<div>
    To calculate the intensity gradient of the grayscale image, I calculated the Sobel gradient magnitude in the (x,y) directions. Then, I applied the DTCWT to generate 6 different highpass subbands roughly corresponding to 15, 45, 75, 105, 135, 165 degrees correspondingly.
</div> <br/>

<div>
    Next, a seperable median filter was applied to each wavelet subband perform nonlinear noise removal. To extract the texture gradient, the Gaussian derivative function was applied in the x and y directions expressed as 
</div>


