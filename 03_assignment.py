'''
Assignment #3

1. Add / modify code ONLY between the marked areas (i.e. "Place code below"). Do not modify or add code elsewhere.
2. Run the associated test harness for a basic check on completeness. A successful run of the test cases does not guarantee accuracy or fulfillment of the requirements. Please do not submit your work if test cases fail.
3. To run unit tests simply use the below command after filling in all of the code:
    python 03_assignment.py
  
4. Unless explicitly stated, please do not import any additional libraries but feel free to use built-in Python packages
5. Submissions must be a Python file and not a notebook file (i.e *.ipynb)
6. Do not use global variables
7. Make sure your work is committed to your master branch in Github
8. Use the test cases to infer requirements wherever feasible


'''
import csv, json, math, pandas as pd, requests, unittest, uuid

# ------ Create your classes here \/ \/ \/ ------

# Box class declaration below here
    
class Box:

    ## Constructor declaration.
    def __init__(self, length, width):
        self.__length = length
        self.__width = width

    ## A method that prints out to the screen a box made with asterisks of length and width dimensions.
    def render(self):
        for i in range(self.__length):
            if i in[0]:
                print("* "*(self.__width))
            elif i in[(self.__length-1)]:
                print("* "*(self.__width))
            else:
                print("*"+"  "*(self.__width-2)+" *")

    ## A method that switches length and width with each other.
    def invert(self):
        self.__length, self.__width = self.__width, self.__length
    
    ## A method that gets area of a box.
    def get_area(self):
        return self.__length * self.__width

    ## A method that get perimeter of a box.
    def get_perimeter(self):
        return self.__length * 2 + self.__width * 2

    ## A method that doubles the size of a box.
    def double(self):
        double = Box(self.__length * 2, self.__width * 2)
        return double

    ## A method that gets length of a box.
    def get_length(self):
        return self.__length

    ## A method that gets width of a box.
    def get_width(self):
        return self.__width

    ## A method that compares two boxes with respect to their length and width.
    def __eq__(self, other):
        if self.__length == other.get_length() and self.__width == other.get_width():
            return True
        else:
            return False 

    ## A method that prints length and width of a box.
    def print_dim(self):
        print('The box length is:', self.__length)
        print('The box width is:', self.__width)

    ## A method that gets dimensions of a box.
    def get_dim(self):
        return self.__length, self.__width

    ## A method that increases the length and width by the dimensions of the box passed in.
    def combine(self, other):
        combine = Box(self.__length + other.get_length(), self.__width + other.get_width())
        return combine
    
    ## A method that finds the length of the diagonal that cuts throught the middle of a box.
    def get_hypot(self):
        return math.sqrt(self.__length**2 + self.__width**2)


# MangoDB class declaration below here

class MangoDB:

    ## Constructor declaration and set default collection.
    def __init__(self):
        self.__db = {'default': {'version':1.0,'db':'mangodb','uuid':str(uuid.uuid4())}}

    ## A method that displays all collections in dictionaries using for loop.
    def display_all_collections(self):
        for k1, v1 in self.__db.items():
            print('Collection:', k1)
            for k2, v2 in v1.items():
                print(k2,v2) 
        
    ## A method that adds new collection name.
    def add_collection(self, collection_name):
        self.__db.update({collection_name:{}})

    ## A method that inserts a new item to a collectiom.
    def update_collection(self, collection_name, updates):
        new_item = self.__db[collection_name]
        new_item.update(updates)

    ## A method that deletes a specific collection by name and its associated data.
    def remove_collection(self, collection_name):
        del self.__db[collection_name]

    ## A method that displays a list of all the collections.
    def list_collections(self):
        for keys in self.__db.keys():
            print(keys)

    ## A method that finds the number of key/value pairs in a given collection.
    def get_collection_size(self, collection_name):
        return len(self.__db.get(collection_name))
    
    ## A method that converts the collection to a JSON string.
    def to_json(self, collection_name):
        return json.dumps(self.__db.get(collection_name), sort_keys=True)

    ## A method that resets to the default collection.
    def wipe(self):
        self.__init__()

    ## A method that returns a list of collection names.
    def get_collection_names(self):
        return list(self.__db.keys())

    ## A method that accesses the value of a key in a given collection.
    def get_value(self, collection_name, key):
        print(self.__db.get(collection_name)[key])

# ------ Create your classes here /\ /\ /\ ------


def exercise01():

    '''
        Create an immutable class Box that has private attributes length and width that takes values for length and width upon construction (instantiation via the constructor). Make sure to use Python 3 semantics. Make sure the length and width attributes are private and accessible only via getters. Immutable here means that any change to its internal state results in a new Box being returned.
        
        Remember, here immutable means there are no setter methods. States can change with the methods required below i.e. combine(), invert().  A state change means that a new Box is created but the original Box remains untouched.
        
        In addition, create...
        - A method called render() that prints out to the screen a box made with asterisks of length and width dimensions
        - A method called invert() that switches length and width with each other
        - Methods get_area() and get_perimeter() that return appropriate geometric calculations
        - A method called double() that doubles the size of the box. Hint: Pay attention to return value here
        - Implement __eq__ so that two boxes can be compared using ==. Two boxes are equal if their respective lengths and widths are identical.
        - A method print_dim that prints to screen the length and width details of the box
        - A method get_dim that returns a tuple containing the length and width of the box
        - A method combine() that takes another box as an argument and increases the length and width by the dimensions of the box passed in
        - A method get_hypot() that finds the length of the diagonal that cuts throught the middle

        In the function exercise01():
        - Instantiate 3 boxes of dimensions 5,10 , 3,4 and 5,10 and assign to variables box1, box2 and box3 respectively 
        - Print dimension info for each using print_dim()
        - Evaluate if box1 == box2, and also evaluate if box1 == box3, print True or False to the screen accordingly
        - Combine box3 into box1 (i.e. box1.combine())
        - Double the size of box2
        - Combine box2 into box1
        - Using a for loop, iterate through and print the tuple received from calling box2's get_dim()
        - Find the size of the diagonal of box2

'''

    # ------ Place code below here \/ \/ \/ ------

    ## Instantiate 3 boxes of dimensions 5,10 , 3,4 and 5,10.
    box1 = Box(5,10)
    box2 = Box(3,4)
    box3 = Box(5,10)
    
    ## Print dimensions for each box.
    box1.print_dim()
    box2.print_dim()
    box3.print_dim()

    ## Evaluate if box1 == box2 and if box1 == box3.
    print(box1==box2)
    print(box1==box3)

    ## Combine box3 into box1.
    box1 = box1.combine(box3)

    ## Double the size of box2.
    box2 = box2.double()

    ## Combine box2 into box1.
    box1 = box1.combine(box2)
           
    ##  Print the box2 dimension through the get_dim() method using a loop.
    for i in box2.get_dim():
        print(i)

    ## Find the size of the diagonal of box2.
    print(box2.get_hypot())

    return box1, box2, box3

    # ------ Place code above here /\ /\ /\ ------


def exercise02():
    '''
    Create a class called MangoDB. The MangoDB class wraps a dictionary of dictionaries. At the the root level, each key/value will be called a collection, similar to the terminology used by MongoDB, an inferior version of MangoDB ;) A collection is a series of 2nd level key/value paries. The root value key is the name of the collection and the value is another dictionary containing arbitrary data for that collection.

    For example:

        {
            'default': {
            'version':1.0,
            'db':'mangodb',
            'uuid':'0fd7575d-d331-41b7-9598-33d6c9a1eae3'
            },
        {
            'temperatures': {
                1: 50,
                2: 100,
                3: 120
            }
        }
    
    The above is a representation of a dictionary of dictionaries. Default and temperatures are dictionaries or collections. The default collection has a series of key/value pairs that make up the collection. The MangoDB class should create only the default collection, as shown, on instantiation including a randomly generated uuid using the uuid4() method and have the following methods:
        - display_all_collections() which iterates through every collection and prints to screen each collection names and the collection's content underneath and may look something like:
            collection: default
                version 1.0
                db mangodb
                uuid 739bd6e8-c458-402d-9f2b-7012594cd741
            collection: temperatures
                1 50
                2 100 
        - add_collection(collection_name) allows the caller to add a new collection by providing a name. The collection will be empty but will have a name.
        - update_collection(collection_name,updates) allows the caller to insert new items into a collection i.e. 
                db = MangoDB()
                db.add_collection('temperatures')
                db.update_collection('temperatures',{1:50,2:100})
        - remove_collection() allows caller to delete a specific collection by name and its associated data
        - list_collections() displays a list of all the collections
        - get_collection_size(collection_name) finds the number of key/value pairs in a given collection
        - to_json(collection_name) that converts the collection to a JSON string
        - wipe() that cleans out the db and resets it with just a default collection
        - get_collection_names() that returns a list of collection names


        Make sure to never expose the underlying data structures

        For exercise02(), perform the following:

        - Create an instance of MangoDB
        - Add a collection called testscores
        - Take the test_scores list and insert it into the testscores collection, providing a sequential key i.e 1,2,3...
        - Display the size of the testscores collection
        - Display a list of collections
        - Display the db's UUID
        - Wipe the database clean
        - Display the db's UUID again, confirming it has changed
    '''

    test_scores = [99,89,88,75,66,92,75,94,88,87,88,68,52]

    # ------ Place code below here \/ \/ \/ ------

    ## Create an instance of MangoDB.
    mb = MangoDB()

    ## Add a collection called testscores.
    mb.add_collection('testscores')

    ## Take the test_scores list and insert it into the testscores collection, providing a sequential key.
    new_test_scores={}
    for i in range(len(test_scores)):
       new_test_scores[i]=test_scores[i]
    mb.update_collection('testscores', new_test_scores)

    ## Display the size of the testscores collection.
    print(mb.get_collection_size('testscores'))

    ## Display a list of collections.
    mb.list_collections()

    ## Display the db's UUID.
    mb.get_value('default', 'uuid')

    ## Wipe the database clean.
    mb.wipe()

    ## Display the db's UUID again, confirming it has changed
    mb.get_value('default', 'uuid')


    # ------ Place code above here /\ /\ /\ ------


def exercise03():
    '''
    1. Avocado toast is expensive but enormously yummy. What's going on with avocado prices? Read about avocado prices (https://www.kaggle.com/neuromusic/avocado-prices/home)
    2. Load the avocado.csv file included in this Githb repository and display every line to the screen
    3. Open the file name under csv_file
    4. The reader should be named reader
    5. Use only the imported csv library to read and print out the avacodo file
    
    '''

    # ------ Place code below here \/ \/ \/ ------

    ## Load the avocado.csv file and converted to csv.reader object.
    with open('C:\\Users\\Sie Siong Wong\\Desktop\\CUNY SPS\\Courses\\Spring 2020\\DATA 602\\Assignments\\pythoncourse-master\\avocado.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)

    ## Display every line to the screen including the column names.
        print(list(reader))

    ## If using the loop method below, it will skip the column names row.

        '''
        for row in reader:
            print(row)
            
        '''
        
    csv_file.close()

    # ------ Place code above here /\ /\ /\ ------

class TestAssignment3(unittest.TestCase):
    def test_exercise01(self):
        print('Testing exercise 1')
        b1, b2, b3 = exercise01()
        self.assertEqual(b1.get_length(),16)
        self.assertEqual(b1.get_width(),28)
        self.assertTrue(b1==Box(16,28))
        self.assertEqual(b2.get_length(),6)
        self.assertEqual(b2.get_width(),8)
        self.assertEqual(b3.get_length(),5)
        self.assertEqual(b2.get_hypot(),10)
        self.assertEqual(b1.double().get_length(),32)
        self.assertEqual(b1.double().get_width(),56)
        self.assertTrue(6 in b2.get_dim())
        self.assertTrue(8 in b2.get_dim())
        self.assertTrue(b2.combine(Box(1,1))==Box(7,9))

    def test_exercise02(self):
        print('Testing exercise 2')
        exercise02()
        db = MangoDB()
        self.assertEqual(db.get_collection_size('default'),3)
        self.assertEqual(len(db.get_collection_names()),1)
        self.assertTrue('default' in db.get_collection_names() )
        db.add_collection('temperatures')
        self.assertTrue('temperatures' in db.get_collection_names() )
        self.assertEqual(len(db.get_collection_names()),2)
        db.update_collection('temperatures',{1:50})
        db.update_collection('temperatures',{2:100})
        self.assertEqual(db.get_collection_size('temperatures'),2)
        self.assertTrue(type(db.to_json('temperatures')) is str)
        self.assertEqual(db.to_json('temperatures'),'{"1": 50, "2": 100}')
        db.wipe()
        self.assertEqual(db.get_collection_size('default'),3)
        self.assertEqual(len(db.get_collection_names()),1)
        
    def test_exercise03(self):
        print('Exercise 3 not tested')
        exercise03()
     

if __name__ == '__main__':
    unittest.main()
