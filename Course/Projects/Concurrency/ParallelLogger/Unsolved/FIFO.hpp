#pragma once
/*!
 * @file FIFO.hpp
 * @date 11 Nov 2012
 * @brief Header file that contains the FIFO class. 
 * 
 * Managing POSIX FIFOs requires keeping track of a lot of components. 
 * to solve this issue the FIFO template serves as an abstraction layer over
 * connecting/creating to a fifo and then provides utilities to pass simple 
 * data structures over the fifo. 
 *
 */
//CPP includes
// Templated classes must contain logic that is contained within the header file. 
#include<string>   ///< Needed for: string
#include<cstring>  ///< Needed for: strerror, errno
#include<thread>   ///< Needed for: thread
#include<cstdio>   ///< Needed for: remove
#include<fcntl.h>  ///< needed for: open
#include<unistd.h> ///< Needed for: close
#include<iostream> ///< Needed for: cout, endl;

//linux system includes
#include<sys/stat.h>

//#ifndef FIFO_HPP
//#define FIFO_HPP

#define APPEND_MODE (O_APPEND | O_WRONLY)
#define DEQUEUE_MODE O_RDONLY
#define BOTH_MODE O_RDWR

namespace project 
{
    //! \class FIFO 
    //! \brief Abstraction layer over linux fifo utilities.  
    //!
    //! Used to hide the complexities of setting up a fifo. 
    template <class T>
    class FIFO {
        private:
            std::string fifo_pathname_;   ///< The path to the FIFO file. 

            std::string mkfifo_errno_;    ///< mkfifo errno store. 
            std::string open_fifo_errno_; ///< open fifo errno store. 
            std::string append_errno_;    ///< last errno from append.
            std::string dequeue_errno_;    ///< last errno from append.

            std::thread open_thread_;     ///< thread for blocking open operation. 

            int fifo_file_descriptor_;    ///< FIFO file descriptor.  
            int result_mkfifo_;           ///< Result when calling mkfifo()
            bool opened_ = false;         ///< Was the file opened?
            mode_t mode_;                 ///< The FIFO mode. 

            /*! \brief Opens the fifo file. 
             * 
             *  Since opening a FIFO is a blocking operation then make the 
             *  function a class method so that it can be passed off to a 
             *  thread. 
             * 
             *  \param[in, out] a FIFO object to open. 
             */
            static void open_fifo(FIFO<T> * fifo, mode_t mode)
            {
                ///< TODO: Implement code that opens a posix fifo.  
            };

        public:
            /*! \brief Default constructor
             * 
             *  Initializes a FIFO using a pathname. 
             * 
             *  \param[in] the pathname for the fifo to create and open. 
             */
            FIFO<T>(const char * fifo_pathname, mode_t mode) {
                fifo_pathname_ = std::string(fifo_pathname);
                mkfifo_errno_ = strerror(errno);
                
                ///< TODO: Insert code that initializes and opens a FIFOs. 
                 
                if(result_mkfifo_ < 0) {
                    mkfifo_errno_ = strerror(errno);
                }
            };
            
            /*! \brief Default destructor. 
             * 
             *  Delete any used resources. 
             * 
             */
            ~FIFO<T>() {
                ///< TODO: Close open FIFOs and terminate any running threads. 
            }
            
            /*! \brief Delete the represented file. 
             *  
             *  Delete the file that FIFO represents. 
             */
            int delete_file() {
                return std::remove(fifo_pathname_.c_str());
            };

            /*! \brief Append to the FIFO. 
             * 
             *  Appends a simple type to the FIFO. Only components within
             *  the type will be appended to the FIFO. For example if T is
             *  a struct then if it contains pointers to the heap then only 
             *  the pointers will be appended. 
             * 
             *  \param[in] the type to append to the fifo.  
             */
            ssize_t append(const T * to_append)
            {
                ///< TODO: Append data to the fifo. 
                ///< NOTE: Use one of the structs contained in server.hpp for your convenience. 
                return 0;
            }
 
            /*! \brief Dequeue from the FIFO. 
             * 
             *  Dequeues a simple type from the FIFO. Suffers the same 
             *  limitations as 
             * 
             *  \sa append 
             *
             *  \param[in] the type to append to the fifo.  
             */
            size_t dequeue(T * container)
            {
                ///< TODO: Dequeue andy data from the fifo. 
                ///< NOTE: Use one of the structs contained in server.hpp for your convenience. 

                return 0;
            }
    };
}

//#endif
