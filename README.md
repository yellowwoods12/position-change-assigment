# position-change-assigment
Position update assignment 


The above directory doesn't contain any model checkpoints. Please download sam_vit.pth checkpoint from https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view and lama checkpoint from https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg (pretrained_models directory). Kindly put the two in the root folder. 

task1.py takes in an input image and a prompt and generates an output image with the target masked object in red. Use the following command to execute task1.py >

  ### python task1.py --image stool.jpeg --prompt "stool" --output_dir results/

where 'image' takes in the image path, 'prompt' takes in the target object label to be masked and 'output_dir' takes in the path of the output directory.

task2.py takes in an input image along with a prompt label and x, y shifts and outputs an image that has the target object shifted by x,y distance. Use the following command to execute task2.py >

  ### python task2.py --image stool.jpeg --prompt "stool" --output_dir results/ --x 10 --y -50

where 'image' takes in the image path, 'prompt' takes in the target object label to be masked, 'output_dir' takes in the path of the output directory and x and y is the distance by which we want to shift the target object specified by the prompt label on the x and y axis respectively from its original position.

## Example Results 

![bagpack](https://github.com/yellowwoods12/position-change-assigment/assets/31931348/d454be9c-b7a1-49a8-abdd-af9758980566) 
![masked_object](https://github.com/yellowwoods12/position-change-assigment/assets/31931348/18da1f06-e986-4dcf-8f03-e3b0b52e8044) 
![bagpack_modified_image](https://github.com/yellowwoods12/position-change-assigment/assets/31931348/9f6a0d93-d26e-4cf7-932a-ef924c3f5c9c)


![stool](https://github.com/yellowwoods12/position-change-assigment/assets/31931348/6d151e8d-4118-4b18-934c-324cc4a12c5a)
![stool_masked_object](https://github.com/yellowwoods12/position-change-assigment/assets/31931348/fb1ad3e8-bc79-42ac-b9d0-c74a9bf02d1c)
![stool_modified_image](https://github.com/yellowwoods12/position-change-assigment/assets/31931348/44396280-4a0f-43bd-8408-3115e1e2c04a)


![wall hanging](https://github.com/yellowwoods12/position-change-assigment/assets/31931348/5cdf6fbd-2cf9-461d-941e-807af943834c)
![wall_hanging_masked_object](https://github.com/yellowwoods12/position-change-assigment/assets/31931348/8f535ba1-a65e-4839-9932-fafe068f59c0)
![wall_hanging_modified_image](https://github.com/yellowwoods12/position-change-assigment/assets/31931348/be5fa2db-106e-4bc5-bec2-f127c6d60a7a)

## Approach 

The approach for this assignment was quite straight-forward. We first employ the [grounding-DINO](https://arxiv.org/pdf/2303.05499) model to detect the bouding box location of the object mentioned in the text prompt in the provided input image. Once we have the x,y,h,w coordinates we understand the region in which that particular object is located. We then use [Segment Anything Model](https://scontent-maa2-1.xx.fbcdn.net/v/t39.2365-6/10000000_900554171201033_1602411987825904100_n.pdf?_nc_cat=100&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=MTpN6mv_LokQ7kNvgFNKP9p&_nc_ht=scontent-maa2-1.xx&oh=00_AYBQCVwTGnFFYr2kOyX8T2b5NELOusdkDguYVlDC5gbJUQ&oe=667ECB67) by passing in  the input image along with the box coordinates of the region of interest to further generate the mask for the desired object. We then make the masked region red in the generated output image and that gives us the results for task 1. For task2, once we have the predicted masked region our next step is to inpaint the masked region from the surrounding pixels inorder to generate an intermediate image without the masked object. For this we use [LaMa](https://github.com/advimman/lama) which is a large mask inpainting model that uses Fourier Convolutions for inpainting a given masked region in an image. Once we generate the intermediate image, our next step is to place the masked object at the location of (prev_x+x_shift,prev_y+y_shift) where x_shift and y_shift are specified as an input. This is simply done by shifting the masked region by the specified (x_shift,y_shift) value and then placing the masked object at this new position of the masked region in the inpainted image. This returns the final image with the updated position of the masked object. 


## Failures, Successes, Improvements

1. One of the major issues that occured during this task is the boundary effect that remains after inpainting. Ideally, we would want a region that blends naturally with its surroundings and does not give the idea of inpainting in some region. This problem is fixed by using the cv2.dilate function to remove sharp boundaries of the masked region and further expand the masked region in such a way that it blends naturally with the rest of the image. To this end, we specify the dilate_kernel_size as a parameter to the dilate_mask() function that essentially passes this variable to the dilate morphological function.
2. Another important aspect that was kept a bit open-ended from the assignment instructions is the shift operation. What should one do if the specified x_shift, y_shift leads towards shifting the mask out of bounds of the image size?
  2a. One solution is to update the x_shift by (prev_x+x_shift % width) and y_shift by (prev_y+y_shift % height) and then perform the object position update.
  2b. Another possible solution is to take the min(prev_x+x_shift, width) and min(prev_y+y_shift, height).
3. Inorder to improve performance, the two tasks can be done back-to-back without saving the intermediate output.

