A Python RPC3 file class for read/write access to time data acquisition files.

.. |br| raw:: html

    <br/>

About RPC III
=============

The RPC III file format, developed by MTS Systems Corporation, is widely used in vehicle durability
testing and simulation. This format is structured as sequential, fixed-length files, with each record
being 512 bytes. The file consists of a standard header followed by data records. The header typically
includes metadata such as the file creation date, channel information, and other configuration details
stored as keyword-value pairs.

The format supports up to 256 data channels and 1024 parameters, with specific limitations on property
names (maximum 32 characters) and values (maximum 96 characters). These files are often used in
conjunction with MTS's RPC Pro and RPC Connect software, which facilitate the analysis and simulation
of vehicle response to road conditions.

To work with RPC III files, tools like the MTS DataPlugin are available, allowing for the reading and
writing of these files within different software environments, such as National Instruments' LabVIEW.
The files typically carry extensions like .rsp or .tim.

If you're looking for a more detailed description of the format or specific documentation, it's usually
included in the software manuals provided by MTS or in the release notes of tools like the MTS


About this module
=================

This module provides functionality to read and write time history data from
RPC III format data files, commonly used in vehicle durability testing.
The module supports the extraction and modification of metadata, such as
channel information and test parameters, as well as the ability to handle
the structured data records stored within these files.

Key Features:
-------------
- Read and parse RPC III (.rsp, .tim) files.
- Write data and metadata back to RPC III files.
- Support for up to 256 data channels and 1024 parameters.
- Handles fixed-length records with standard headers.

Typical usage example:
----------------------
.. code-block:: python

    from scipy import integrate
    import rpc3

    # Read data from an RPC III file
    filename = "samplefile.rpc"
    data = rpc3.to_dict(rpc3.read(filename)[0])

    # Modify data or metadata as needed
    dt = data["V_CGES"].dt
    t = data["V_CGES"].time()
    data["V_CGES"].data /= 3.6
    data["V_CGES"].unit = "m/s"
    data["new_channel"] = (
        rpc3.Channel(
            name="distance",
            unit="m",
            data=integrate.cumulative_trapezoid(data["V_CGES"].data, dx=dt),
            dt=dt
        )
    )

    # Write the modified data back to an RPC III file
    rpc3.write('modified_data.rsp', list(data.values()))


RPC III File Format Summary
===========================

The RPC III file format is used to read, write, and import data in various file types related to time history,
histograms, and other data. The format is structured around sequential, fixed-length, 512-byte records and
includes a standard header followed by data blocks. There is no end-of-file character in these binary files,
except for configuration files, which only contain a header.

File Headers
------------
RPC III file headers provide essential metadata about the data contained within the file.
These headers consist of ASCII records stored in 512-byte blocks, with each record containing a keyword-value
pair that is 128 bytes long. Keywords can be up to 32 bytes, and values can be up to 96 bytes, both including
null terminators. These headers are crucial for RPC III programs, which use byte arrays for string
manipulation in both C and FORTRAN environments.

The first three records in any RPC III file header—FORMAT, NUM_HEADER_BLOCKS, and NUM_PARAMS—are
position-sensitive, meaning they must appear in specific positions. Subsequent records can be placed
arbitrarily, though certain descriptors should be grouped for faster access. The number of header records
is not fixed and can be expanded as needed, limited only by available disk space.

.. figure:: img/file_structure.png
    :align: center

    File organisation with header and data section

.. figure:: img/key_values.png
    :align: center

    Header consists of key value pairs of fixed size each

.. figure:: img/names.png
    :align: center

    Names are stored in C-style (nul-terminated)

File Types
----------
RPC III supports various file types, including TIME_HISTORY, CONFIGURATION (a subset of TIME_HISTORY), MATRIX,
FATIGUE, ROAD_SURFACE, SPECTRAL, and START. Each file type can contain multiple kinds of files, identified
by specific keyword-value pairs in the header. For instance, different kinds of time histories within a
TIME_HISTORY file are distinguished by the TIME_TYPE keyword.
In this module, only TIME_HISTORY datasets are supported.

Data Organization
-----------------
In TIME_HISTORY files, data is organized into groups and frames. Frames can contain between 256 and 16,384 data
points, while groups range from 2,048 to 16,384 data points. Data is demultiplexed, meaning that each block of
data contains sequential data points for multiple channels. The last block in a data sequence may not be
completely filled, with any remaining points set to zero. Normally, the remaining points are are filled with
the last value of the actual time series, so that when reading back the file it is easier to distinguish between
real zero values and zeros that are used to fill the end gap.

This summary provides an overview of the RPC III file format, including its organization, header structure,
and data management. This format is designed to be flexible and expandable, accommodating various data types
and storage requirements.

.. figure:: img/multiplex.png
    :align: center

    Channel data is arranged multiplexed

.. figure:: img/multiplex_2.png
    :align: center

    This figure shows an example with zero padded final frame

Header format with keys describing a TIME_HISTORY dataset
---------------------------------------------------------

+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **Keyword**           | **Description**                             | **Possible Values**                                         |
+=======================+=============================================+=============================================================+
| **BYPASS_FILTER**     | Turns A/D filtering on or off.              | **Options:** |br|                                           |
|                       |                                             | 0: Uses the filter (BYPASS_FILTER is off) |br|              |
|                       |                                             | 1: Does not use the filter (BYPASS_FILTER is on)            |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **CHANNELS**          | Number of channels in the file.             | Range of channels is 1 to 128.                              |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **DATA_TYPE**         | Numbers contained in the data of the file   | **Data types are:**  |br|                                   |
|                       | are of this type.                           | - SHORT_INTEGER (Default used by all RPC III programs) |br| |
|                       |                                             | - FLOATING_POINT (Not supported by RPC III V5.0x programs)  |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **DATE**              | Date and time the file or file header       | NTFS date and time. |br|                                    |
|                       | was created.                                | **Example:** 22-Feb-2000  10:20:11 |br|                     |
|                       |                                             | **Format:** dd-mm-yyyy hh:mm:ss                             |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **DELTA_T**           | Time interval between consecutive points    | A real number. |br|                                         |
|                       | of data (in seconds).                       | **Example:** 4.882812E-03 |br|                              |
|                       |                                             | **Format:** E8.6 |br|                                       |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **DESC.CHAN_n**       | ASCII description of the specified          | Enclose multi-word descriptions in quotation                |
|                       | channel. This description is repeated       | marks (" "). |br|                                           |
|                       | for each channel in the file.               | Max 96 chars for RPC III, 20 for RPC II. |br|               |
|                       |                                             | **Example:** "Left longitudinal axis"                       |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **FILE_TYPE**         | Type of data file.                          | File type is: |br|                                          |
|                       |                                             | TIME_HISTORY                                                |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **FORMAT**            | Format in which data is stored.             | Data storage formats are:  |br|                             |
|                       |                                             | - BINARY_IEEE_LITTLE_END  |br|                              |
|                       |                                             | - BINARY_IEEE_BIG_END  |br|                                 |
|                       |                                             | - BINARY  |br|                                              |
|                       |                                             | - ASCII  |br|                                               |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **FRAMES**            | Number of frames of data stored in a file.  | Any valid number of frames.                                 |
|                       | A frame is a set of data points ranging     |                                                             |
|                       | from 2 to any integer that is a power of 2. |                                                             |
|                       | Max number of points in a frame is 8192.    |                                                             |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **HALF_FRAMES**       | Specifies whether a half frame is added to  | **Options:**  |br|                                          |
|                       | the beginning and end of a file to allow    | 0: No half frames added |br|                                |
|                       | some forms of mathematical processing.      | 1: Half frames added                                        |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **INT_FULL_SCALE**    | The maximum 16 bit integer value of the     | Default value is 32752 (which is 2^16 - 16)                 |
|                       | data                                        |                                                             |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **LOWER_LIMIT.CHAN_n**| Lower limit value defined for channel n.    | Any valid number.                                           |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **MAP.CHAN_n**        | Physical channel to which logical channel   | Any valid physical channel number. Channel mapping may      |
|                       | is mapped.                                  | change for multi-rate responses or multi-test setups.       |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **NUM_HEADER_BLOCKS** | Number of 512-byte blocks used by the       | Up to 256 blocks. |br|                                      |
|                       | header information.                         | **Example:** 9                                              |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **NUM_PARAMS**        | Total number of parameters (keyword-value   | Maximum number of parameters is 1024 (256 * 4).             |
|                       | pairs) in the file header.                  |                                                             |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **OPERATION**         | Name of program or operation that created   | Any RPC III software program or operation. |br|             |
|                       | the file.                                   | **Example:** SINESW                                         |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **PARENT_k**          | File(s) from which this file was created.   | Any valid NTFS file name(s). |br|                           |
|                       | (k represents number of file.)              | **Example:** C:/TEST/DATA/POT.RSP                           |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **PART.CHAN_n**       | First channel assigned to partition n.      | Any valid integer. Default value is 1.                      |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **PART.NCHAN_n**      | Number of consecutive channels assigned     | Any valid integer. Default value is the value in the        |
|                       | to partition n.                             | keyword CHANNELS.                                           |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **PARTITIONS**        | The number of groups of channels wanted.    | Range is 1 to 128 partitions. Default value is 1. |br|      |
|                       | A partition is a submatrix within a         | Specify partitions if numerical operations on a time        |
|                       | matrix file.                                | history data file will create a matrix file.                |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **PTS_PER_FRAME**     | Integer that indicates number of points     | **Options are:** 256, 512, 1024, and 2048 points.           |
|                       | per frame of stored data. Must be a power   | It is a function of PTS_PER_GROUP.                          |
|                       | of 2.                                       | PTS_PER_FRAME must be ≤ PTS_PER_GROUP.                      |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **PTS_PER_GROUP**     | Total number of data points in the group.   | Integer sum of all points. |br|                             |
|                       | A group is a set of 2^n data frames in a    | **Example:** |br|                                           |
|                       | channel.                                    | If there are 4 frames in a group, then PTS_PER_GROUP        |
|                       |                                             | is the sum of the points of those four frames: |br|         |
|                       |                                             | Frame 1 = 2048 points |br|                                  |
|                       |                                             | Frame 2 = 2048 points |br|                                  |
|                       |                                             | Frame 3 = 2048 points |br|                                  |
|                       |                                             | Frame 4 = 2048 points |br|                                  |
|                       |                                             | Points per group = 8192 points                              |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+
| **REPEATS**           | Number of times the frame is identically    | Typically 1 repeat.                                         |
|                       | repeated within the file the first time     |                                                             |
|                       | the file is played out.                     |                                                             |
+-----------------------+---------------------------------------------+-------------------------------------------------------------+

References
----------

- `National Instruments MTS DataPlugin for RPC-III <https://www.ni.com>`_
- `MTS RPC Connect Software <https://www.mts.com>`_
- `GitHub RPC3.js Viewer <https://github.com/galuszkm/RPC3.js>`_
- `RPC3 file format <https://github.com/galuszkm/RPC3.js/blob/master/public/RPC3_Format.pdf>`_

Documentation generated with the assistance of ChatGPT.

.. codeauthor:: Andreas Martin

.. list-table::

    * - *Date:*
      - 2023-04-05
    * - *Requires:*
      - Python>=3.7, numpy>=1.19, tqdm
