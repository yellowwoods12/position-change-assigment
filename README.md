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

