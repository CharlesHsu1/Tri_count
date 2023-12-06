from mpi4py import MPI
import pandas as pd
import os

def calculate_and_save_degree(data, directory, rank):
    degrees = data['FromNodeID'].value_counts().reset_index()
    degrees.columns = ['Node', 'Degree']
    degrees.to_csv('{}/degree_rank_{}.csv'.format(directory, rank), index=False)

def save_node_links(data, directory, rank):
    links = data.groupby('FromNodeID')['ToNodeID'].apply(list).reset_index()
    links.columns = ['Node', 'Links']
    links.to_csv('{}/links_rank_{}.csv'.format(directory, rank), index=False)

def create_and_save_edge_index(data, directory, rank):
    edges = data.apply(lambda row: tuple(sorted([row['FromNodeID'], row['ToNodeID']])), axis=1)
    edge_index = pd.DataFrame(edges.unique(), columns=['Edge'])
    edge_index.to_csv('{}/edge_index_rank_{}.csv'.format(directory, rank), index=False)

def save_nodes(data, directory, rank):
    nodes_from = pd.DataFrame(data['FromNodeID'].unique(), columns=['Node'])
    nodes_to = pd.DataFrame(data['ToNodeID'].unique(), columns=['Node'])
    all_nodes = pd.concat([nodes_from, nodes_to]).drop_duplicates().reset_index(drop=True)
    all_nodes.to_csv('{}/nodes_rank_{}.csv'.format(directory, rank), index=False)    

def aggregate_results(temp_directory, output_directory, size):
    for file_type in ['degree', 'links', 'edge_index', 'nodes']:
        all_data = []
        for i in range(size):
            file_path = '{}/{}_rank_{}.csv'.format(temp_directory, file_type, i)
            all_data.append(pd.read_csv(file_path))

        combined_data = pd.concat(all_data)

        if file_type == 'degree':
            combined_data = combined_data.groupby('Node')['Degree'].sum().reset_index()
        elif file_type == 'edge_index':
            combined_data = combined_data.drop_duplicates().reset_index(drop=True)
        elif file_type == 'links':
            combined_data = combined_data.groupby('Node')['Links'].sum().reset_index()
        elif file_type == 'nodes':
            combined_data = pd.DataFrame(combined_data['Node'].unique(), columns=['Node'])

        combined_data.to_csv('{}/{}.csv'.format(output_directory, file_type), index=False)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    input_file_path = '/DataPath/chunk_{}.csv'.format(rank)
    temp_directory = '/DataPath/temp_data/'
    output_directory = '/DataPath/tri_count/'

    if rank == 0:
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    comm.Barrier()

    data = pd.read_csv(input_file_path)
    
    calculate_and_save_degree(data, temp_directory, rank)
    save_node_links(data, temp_directory, rank)
    create_and_save_edge_index(data, temp_directory, rank)
    save_nodes(data, temp_directory, rank)

    if rank == 0:
        aggregate_results(temp_directory, output_directory, size)

    if rank == 0:
        for file in os.listdir(temp_directory):
            os.remove(os.path.join(temp_directory, file))

    if rank == 0:
        total_nodes = pd.read_csv('{}/nodes.csv'.format(output_directory))['Node'].nunique()
        total_edges = pd.read_csv('{}/edge_index.csv'.format(output_directory))['Edge'].count()
        print("Total nodes: {}, Total edges: {}".format(total_nodes, total_edges))

if __name__ == '__main__':
    main()

