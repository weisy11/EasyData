# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from . import classification
from . import detection
from . import connector
from . import base

from .classification import *
from .detection import *
from .visual import *
from .connector import *
from .keypoint import *
from .base import *

__all__ = classification.__all__ + detection.__all__ + visual.__all__ + connector.__all__ + keypoint.__all__ + base.__all__
