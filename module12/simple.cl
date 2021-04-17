//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void square(__global *buffer)
{
    size_t id = get_global_id(0);

    buffer[id] = buffer[id] * buffer[id];
}

__kernel void average(__global int *buffer, int size)
{
    size_t id = get_global_id(0);

    int start = id * size;
    int end = start + size;

    int average = 0;

    for (size_t i = start; i < end; i++)
    {
        average += buffer[i];
    }

    average /= size;

    for (size_t i = start; i < end; i++)
    {
        buffer[i] = average;
    }
}