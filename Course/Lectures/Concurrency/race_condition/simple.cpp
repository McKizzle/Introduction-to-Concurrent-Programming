void handle_request(int & requests_served) {
    requests_served++;
}

int main(int argc, char * argv[]) {
    int requests_served = 0;
    handle_request(requests_served);
    return 0;
}
